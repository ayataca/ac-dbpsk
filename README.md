# acoustic-communication-dbpsk

### 「可聴域と近非可聴域を併用した音響通信」

- DBPSK.ipynb
    - 差動ビット位相偏移変調(DBPSK)の「エンコード→録音→デコード」の一連の流れ
    - 時間軸上で一つ前の位相からπずれているか，何もずれていないかで1/0を表現
    - 最初のallocate_subcarriersでサブキャリアの位置を指定
- realtime_demo
    - 音源が既知の場合のリアルタイムデコード
    - 音源はコマンドライン引数で指定する
