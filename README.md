# acoustic-communication-dbpsk

### 「可聴域と近非可聴域を併用した音響通信」
井上景太

- DBPSK.ipynb
    - 差動ビット位相偏移変調(DBPSK)の「エンコード→録音→デコード」の一連の流れ
    - 最初のallocate_subcarriersでサブキャリアの位置を指定
- realtime_demo
    - 音源が既知の場合のリアルタイムデコード
    - 音源はコマンドライン引数で指定する
        - エンコード  
          ```
          python3 encode.py 音源名 埋め込む文字列(10文字以内)
          ```
        - デコード  
          ```
          python3 decode.py 音源名
          ```

- 実験の様子
<img src="https://github.com/user-attachments/assets/4ca5f0ca-42a6-46c0-8baa-8fc3e9e8d913" alt="実験の様子" width="400">

- 本研究の
1. 研究内容をまとめたポスター
2. 卒論原稿
3. 卒論発表スライド  
が入った[Google Driveのフォルダ](https://drive.google.com/drive/folders/19DHQOeJDA0TOYHQsiJZgbRhEIoTiKT6m?usp=sharing)

