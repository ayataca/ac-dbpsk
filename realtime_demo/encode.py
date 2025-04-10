import sys

import numpy as np
import soundfile as sf

WHITE = "\033[37m"
RED = "\033[31m"
BLUE = "\033[34m"
RESET = "\033[0m"

samplerate = 44100
frame_length = 2048
subcarrier_width = 1000
freqs = np.fft.fftfreq(frame_length, 1 / 44100)
num_frame = 860


def string_to_20x4_bits(s):
    if len(s) != 10:
        raise ValueError(f"入力の文字列が10文字ではありません  文字数:{len(s)}")
    # 文字列をビット列に変換
    bit_list = [bit for char in s for bit in format(ord(char), "08b")]
    # ビット列を4ビットずつ区切り、20×4のリストに変換
    bit_matrix = [bit_list[i : i + 4] for i in range(0, len(bit_list), 4)]
    # 各ビットを整数に変換
    bit_matrix = [[int(bit) for bit in row] for row in bit_matrix]
    return bit_matrix


def allocate_subcarriers(f: float):
    # 利用周波数帯域の定義
    bands = [(1000, 5000, "audible"), (18000, 22000, "near_inaudible")]  # (下限, 上限, フラグ) のリスト

    # サブキャリアの中心周波数を格納するリスト
    subcarriers = []

    for band_start, band_end, flag in bands:
        # この帯域での中心周波数リストを生成
        band_subcarriers = [(band_start + f / 2 + i * f, flag, i) for i in range(int((band_end - band_start) / f))]
        subcarriers.extend(band_subcarriers)

    return len(subcarriers), subcarriers


def DBPSK(X, X_former, bits, subcarrier_list, idx):
    if idx != 50:
        for i in range(len(subcarrier_list)):
            bit = bits[subcarrier_list[i][2]]
            if subcarrier_list[i][1] == "near_inaudible":
                # 近非可聴域は振幅を立てる
                center = int(frame_length * (subcarrier_list[i][0] / 44100))
                for j in range(23):
                    nearly_inaudible_amp = 5
                    if bit == 1:
                        # 1のとき（πずらす）
                        X[center + j] = nearly_inaudible_amp * np.exp(1j * (np.angle(X_former[center + j]) + np.pi))
                        X[center - j] = nearly_inaudible_amp * np.exp(1j * (np.angle(X_former[center - j]) + np.pi))
                        X[frame_length - center - j] = nearly_inaudible_amp * np.exp(
                            -1j * (np.angle(X_former[center + j]) + np.pi)
                        )
                        X[frame_length - center + j] = nearly_inaudible_amp * np.exp(
                            -1j * (np.angle(X_former[center - j]) + np.pi)
                        )
                    else:
                        # 0のとき（何もずらさない）
                        X[center + j] = nearly_inaudible_amp * np.exp(1j * (np.angle(X_former[center + j])))
                        X[center - j] = nearly_inaudible_amp * np.exp(1j * (np.angle(X_former[center - j])))
                        X[frame_length - center - j] = nearly_inaudible_amp * np.exp(-1j * (np.angle(X_former[center + j])))
                        X[frame_length - center + j] = nearly_inaudible_amp * np.exp(-1j * (np.angle(X_former[center - j])))
            elif subcarrier_list[i][1] == "audible":
                # 可聴域は振幅はそのまま
                center = int(frame_length * (subcarrier_list[i][0] / 44100))
                for j in range(23):
                    if bit == 1:
                        # 1のとき（πずらす）
                        X[center + j] = np.abs(X[center + j]) * np.exp(1j * (np.angle(X_former[center + j]) + np.pi))
                        X[center - j] = np.abs(X[center - j]) * np.exp(1j * (np.angle(X_former[center - j]) + np.pi))
                        X[frame_length - center - j] = np.abs(X[center + j]) * np.exp(
                            -1j * (np.angle(X_former[center + j]) + np.pi)
                        )
                        X[frame_length - center + j] = np.abs(X[center - j]) * np.exp(
                            -1j * (np.angle(X_former[center - j]) + np.pi)
                        )
                    else:
                        # 0のとき（何もずらさない）
                        X[center + j] = np.abs(X[center + j]) * np.exp(1j * (np.angle(X_former[center + j])))
                        X[center - j] = np.abs(X[center - j]) * np.exp(1j * (np.angle(X_former[center - j])))
                        X[frame_length - center - j] = np.abs(X[center + j]) * np.exp(-1j * (np.angle(X_former[center + j])))
                        X[frame_length - center + j] = np.abs(X[center - j]) * np.exp(-1j * (np.angle(X_former[center - j])))
    return X


def dpsk_encode(original_signal, bits):
    # サブキャリアの位置
    _, subcarrier_list = allocate_subcarriers(subcarrier_width)

    # ハニング窓
    hanning_window = np.hanning(frame_length)

    # 新しいデータ配列を初期化
    s3_signal = np.zeros(num_frame * frame_length)
    s1_signal_mod = np.zeros(num_frame * frame_length)
    s2_signal = np.zeros(num_frame * frame_length)

    # 一つ前の周波数領域情報
    s1_frame_fft_former = np.zeros(frame_length)

    # 50フレームまではそのまま
    s3_signal[: 50 * frame_length] = original_signal[: 50 * frame_length]

    # 各フレームでFFTを行い、位相を変更
    for i in range(50, num_frame - 1):
        start = i * frame_length
        end = start + frame_length
        s1_frame = hanning_window * original_signal[start:end]
        s2_frame = hanning_window * original_signal[start + frame_length // 2 : end + frame_length // 2]

        s1_frame_fft = np.fft.fft(s1_frame)

        if i == 50:
            s1_frame_fft_former = np.copy(s1_frame_fft)

        char_idx = (i - 50) % 20

        s1_frame_fft_mod = DBPSK(s1_frame_fft, s1_frame_fft_former, bits[char_idx], subcarrier_list, i)

        s1_frame_mod = np.fft.ifft(s1_frame_fft_mod)
        s1_frame_fft_former = np.copy(s1_frame_fft_mod)

        for j in range(frame_length):
            s3_signal[start + j] += s1_frame_mod[j].real
            s1_signal_mod[start + j] += s1_frame_mod[j].real
            if i < num_frame:
                s3_signal[start + frame_length // 2 + j] += s2_frame[j].real
                s2_signal[start + frame_length // 2 + j] += s2_frame[j].real

    return s3_signal


def main():
    args = sys.argv

    input_filename = "rock.wav"
    # match args[1]:
    #     case "rock":
    #         input_filename = "rock.wav"
    #     case "rock_2":
    #         input_filename = "inst.wav"
    #     case "speech":
    #         input_filename = "speech.wav"
    #     case "speech_2":
    #         input_filename = "speech_2.wav"
    #     case "jazz":
    #         input_filename = "flamenco2.wav"
    #     case "silent":
    #         input_filename = "speech_silent.wav"
    if args[1] == "rock":
        input_filename = "rock.wav"
    elif args[1] == "rock_2":
        input_filename = "inst.wav"
    elif args[1] == "speech":
        input_filename = "speech.wav"
    elif args[1] == "speech_2":
        input_filename = "speech_2.wav"
    elif args[1] == "jazz":
        input_filename = "flamenco2.wav"
    elif args[1] == "silent":
        input_filename = "speech_silent.wav"

    original_signal, _ = sf.read(input_filename)

    if original_signal.ndim > 1:
        original_signal = original_signal.mean(axis=1)

    original_signal = original_signal / np.max(np.abs(original_signal))

    input_sentence = args[2]
    if len(input_sentence) < 10:
        input_sentence = f"{input_sentence:<10}"
    elif len(input_sentence) > 10:
        print(f"{RED}入力が10文字を超えています。{RESET}")
        return 0

    print(f"[INFO] {RED}{input_sentence}{RESET} has been encoded.")

    bits = string_to_20x4_bits(input_sentence)

    signal_modulated = dpsk_encode(original_signal, bits)

    output_filename = "demo.wav"
    sf.write(output_filename, signal_modulated, samplerate)


if __name__ == "__main__":
    main()
