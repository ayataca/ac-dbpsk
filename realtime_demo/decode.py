import queue
import sys
import threading
import time

import numpy as np
import pyaudio
import soundfile as sf

WHITE = "\033[37m"
RESET = "\033[0m"

# BRIGHT COLOR
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"

samplerate = 44100
frame_length = 2048
subcarrier_width = 1000
freqs = np.fft.fftfreq(frame_length, 1 / 44100)
# ハニング窓
hanning_window = np.hanning(frame_length)


RATE = 44100
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 4096

BLOCK_SIZE = 40960


audio_queue = queue.Queue()  # 録音スレッド(Producer) → デコードスレッド(Consumer)間のデータ受け渡し
stop_event = threading.Event()  # スレッド停止フラグ


def pyaudio_callback(in_data, frame_count, time_info, status_flags):
    """
    録音コールバック。マイクから取得した音声をキューに追加。
    """
    if stop_event.is_set():
        # 停止指示が出ている場合は録音終了
        return (None, pyaudio.paComplete)

    # キューに生バイナリデータを入れる
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)


def audio_recording_thread():
    """
    PyAudioを使ってマイク録音を行い、コールバックでデータをキューに詰める。
    stop_eventが立つまでループ。
    """
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, stream_callback=pyaudio_callback
    )
    stream.start_stream()

    while not stop_event.is_set():
        time.sleep(0.1)

    # 終了処理
    stream.stop_stream()
    stream.close()
    p.terminate()


def find_start_offset(recorded, original_start):
    # float32 に変換
    rec_f32 = recorded.astype(np.float32)
    orig_f32 = original_start.astype(np.float32)

    correlation = np.correlate(rec_f32, orig_f32, mode="valid")

    max_idx = np.argmax(correlation)
    max_val = correlation[max_idx]

    return max_idx, max_val, correlation


def bits_to_string(bit_matrix):
    # if len(bit_matrix) != 20:
    #     raise ValueError(f"入力の行数が20ではありません  行数:{len(bit_matrix)}")
    # for row in bit_matrix:
    #     if len(row) != 4:
    #         raise ValueError("入力の列数が4ではない行があります")

    # 2次元リストを1次元のビット列に変換
    bit_list = [str(bit) for row in bit_matrix for bit in row]

    # 8ビットごとに区切り、文字に変換
    chars = []
    for i in range(0, len(bit_list), 8):
        byte_str = "".join(bit_list[i : i + 8])
        char_code = int(byte_str, 2)  # 2進数を整数に変換

        try:
            char = chr(char_code)
            # 文字が表示可能か判定（カテゴリが "Cc" や "Cs" のものは不適切）
            if char.isprintable():
                chars.append(char)
            else:
                chars.append(" ")  # 不適切な場合は半角スペース
        except ValueError:
            chars.append(" ")  # 変換不可能な場合も半角スペース

    return "".join(chars)


def allocate_subcarriers(f: float):
    # 利用周波数帯域の定義
    bands = [(5000, 9000, "audible"), (18000, 22000, "near_inaudible")]  # (下限, 上限, フラグ) のリスト

    # サブキャリアの中心周波数を格納するリスト
    subcarriers = []

    for band_start, band_end, flag in bands:
        # この帯域での中心周波数リストを生成
        band_subcarriers = [(band_start + f / 2 + i * f, flag, i) for i in range(int((band_end - band_start) / f))]
        subcarriers.extend(band_subcarriers)

    return len(subcarriers), subcarriers


_, subcarrier_list = allocate_subcarriers(subcarrier_width)


def decode_dpsk_40960(data, previous_frame_fft):
    bits_to_decode_audible = [[0] * 4 for _ in range(20)]
    bits_to_decode_inaudible = [[0] * 4 for _ in range(20)]
    # 一つ前の周波数領域情報
    frame_fft_former = previous_frame_fft

    for i in range(20):
        decode_frame = hanning_window * data[frame_length * i : frame_length * (i + 1)]
        frame_fft = np.fft.fft(decode_frame)

        for k in range(len(subcarrier_list)):
            center = int(frame_length * (subcarrier_list[k][0] / 44100))
            S = 0
            for j in range(45):
                frame_fft_angle = np.angle(frame_fft[center - 22 + j])
                frame_fft_former_angle = np.angle(frame_fft_former[center - 22 + j])
                d = abs(frame_fft_angle - frame_fft_former_angle)
                if np.pi < d < 2 * np.pi:
                    d = 2 * np.pi - d
                S += d
            if S / 45 > np.pi / 2:
                if subcarrier_list[k][1] == "audible":
                    bits_to_decode_audible[i][subcarrier_list[k][2]] = 1
                else:
                    bits_to_decode_inaudible[i][subcarrier_list[k][2]] = 1
            else:
                if subcarrier_list[k][1] == "audible":
                    bits_to_decode_audible[i][subcarrier_list[k][2]] = 0
                else:
                    bits_to_decode_inaudible[i][subcarrier_list[k][2]] = 0
        frame_fft_former = np.copy(frame_fft)

    last_frame_fft = frame_fft
    return bits_to_string(bits_to_decode_audible).ljust(10), bits_to_string(bits_to_decode_inaudible).ljust(10), last_frame_fft


def decoding_thread(input_filename):
    """
    1) キューから取り出した録音データを audio_buffer に蓄積。
    2) 最初だけ相関で音源開始位置 (start_offset) を検出。
    3) 音源開始位置が見つかったら，シンボル開始位置(102400)まで待ち，来たらデコードスタート。
    4) デコードがスタートしたら、1.5秒ごと(=66150サンプルごと)に取り出し，可聴域・近非可聴域それぞれ10文字にデコード。
    """

    # 元音源を読み込み
    signal_original, _ = sf.read(input_filename)
    if signal_original.ndim > 1:
        signal_original = signal_original.mean(axis=1)
    signal_original_start = signal_original[:40000]

    audio_buffer = np.array([], dtype=np.int16)  # 録音データを貯めるバッファ
    found_alignment = False
    decode_start = False
    start_offset = None
    print_mode = False
    block_index = 0  # 何個目のブロックか

    # 一つ前の周波数領域情報
    last_frame_fft = np.zeros(frame_length)

    print(f"[INFO] {YELLOW}Recording...{RESET}")

    while not stop_event.is_set():
        # 1) キューからデータを取り出しバッファに連結
        try:
            # print("バッファの追加")
            data = audio_queue.get(timeout=0.1)
            chunk_array = np.frombuffer(data, dtype=np.int16)
            audio_buffer = np.concatenate([audio_buffer, chunk_array])
        except queue.Empty:
            # print("スルー")
            pass  # キューが空のときはスルー

        # print("len(audio_buffer): ", len(audio_buffer))

        # 2) 音源開始位置特定のため相関を取る
        if not found_alignment:
            if len(audio_buffer) >= 100000:
                offset, max_val, correlation = find_start_offset(audio_buffer, signal_original_start)
                found_alignment = True
                start_offset = offset
                audio_buffer = audio_buffer[start_offset:]
                np.savetxt("correlation.txt", correlation)
                print(f"[INFO] {YELLOW}Audio source detected{RESET}")
        else:
            # 3) 音源開始位置が見つかったら，シンボル開始位置(102400)まで待ち，来たらデコードスタート。
            if not decode_start:
                if len(audio_buffer) >= 102400:
                    audio_buffer = audio_buffer[102400:]
                    decode_start = True
                    print(f"[INFO] {YELLOW}Decoding...{RESET}")
                    print(f"[DECODE]     {RED}可聴域{RESET}     {CYAN}近非可聴域{RESET}")
            else:
                # 4) デコードがスタートしたら、1.5秒ごと(=66150サンプルごと)にデータが来る想定。
                if BLOCK_SIZE <= len(audio_buffer):
                    # 40960サンプル分取り出してデコード
                    data = audio_buffer[:BLOCK_SIZE]
                    data_float = data.astype(np.float32)

                    decoded_text_audible, decoded_text_inaudible, previous_frame_fft = decode_dpsk_40960(
                        data_float, last_frame_fft
                    )
                    last_frame_fft = previous_frame_fft

                    if not print_mode:
                        print_mode = True
                    else:
                        print(f"[DECODE]   {RED}{decoded_text_audible}{RESET}   {CYAN}{decoded_text_inaudible}{RESET}")

                    # このブロックに相当するサンプルは用済みなので、バッファから切り落としたい
                    # print("ブロックを削除")
                    audio_buffer = audio_buffer[BLOCK_SIZE:]
                    block_index += 1

        time.sleep(0.001)


def main():
    args = sys.argv

    input_filename = "rock.wav"
    match args[1]:
        case "rock":
            input_filename = "rock.wav"
        case "rock_2":
            input_filename = "inst.wav"
        case "speech":
            input_filename = "speech.wav"
        case "speech_2":
            input_filename = "speech_2.wav"
        case "jazz":
            input_filename = "flamenco2.wav"
        case "silent":
            input_filename = "speech_silent.wav"

    # スレッドを作成
    recorder_thread_ = threading.Thread(target=audio_recording_thread, daemon=True)
    decoder_thread_ = threading.Thread(target=decoding_thread, args=(input_filename,), daemon=True)

    # スレッド起動
    recorder_thread_.start()
    decoder_thread_.start()

    print("Recording & Decoding threads started. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"[INFO] {YELLOW}Recording&Decoding finished{RESET}")
        stop_event.set()
        recorder_thread_.join()
        decoder_thread_.join()


if __name__ == "__main__":
    main()
