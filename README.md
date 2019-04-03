# Requirement
- keras 2.0.3
- tensorflow 1.12.0
- opencv2

# コード概説
ー main.py<br/>
実行用コード。DEEPEC、TempEC、TempEC-HPの３種類のネットワークの学習・テストができます。<br/>
ー data_utils.py<br/>
データセットの画像を読み込むためのクラスを記述。<br/>
ー NETWORK.py<br/>
それぞれのネットワーク構造を記述。<br/>
<br/>
---以下はデータ収集用のコードなのでシステムを動かすのには無関係。<br/>
ー labelling.py<br/>
動画にアイコンタクトのGround Truthをつける。<br/>
ー detect_face.py<br/>
動画から顔画像、目画像を切り出す。<br/>



# 実行 
基本的にはターミナルで<br/>
「python main.py (エポック数) (テストデータの名前) (実行タイプ)」<br/>
を実行すれば動きます。<br/>

例：python main.py 20 Avec test_cnn  ※CNNを動画Avec.mp4の画像で20エポック学習する場合<br/>

テストデータの名前は、学習時にはそれを除いたデータで学習、テスト時はそのデータで学習、という意味で指定します。<br/>

検出対象の画像はdataset/imagesに動画ごとで別々に入っています。<br/>

学習結果の重みファイルはcacheディレクトリに.h5で保存されます。場所は「ネットワークのタイプ/テストデータの名前/（lstmではタイプステップ数）」。重みファイルは実行のたびに上書きされるので必要に応じて別所に移してください。<br/>

テスト時の結果はresultディレクトリに保存されます。cacheと同じディレクトリ構造です。<br/>
結果として保存されるのは以下の内容です。<br/>
ー ROC.png : ROC曲線<br/>
ー result.csv : ネットワークの出力尤度を並べたもの（出力結果を可視化するときはこれを使う。）<br/>
ー result.txt : 出力結果を分析したもの。具体的には識別ミスの数や、正解率・再現率・F値などがテキストでまとめられる。<br/>
ー imgディレクトリ： 識別ミスした画像が保存される。False-NegativeがFNディレクトリに、False-PositiveがFPディレクトリに保存される。（ただし左目のみ）<br/>

##実行タイプ<br/>
ー train_cnn  : CNN(DEEPEC)の学習<br/>
ー test_cnn  : CNN(DEEPEC)のテスト<br/>
ー train_lstm  : LSTM(TempEC)の学習<br/>
ー test_lstm  : LSTM(TempEC)のテスト<br/>
ー train_lstm_face  : LSTM+face(TempEC-HP)の学習<br/>
ー test_lstm_face  : LSTM+face(TempEC-HP)のテスト<br/>

DEEPECはシングルフレームからのアイコンタクト検出、TempEC・TempEC-HPは連続した複数フレームからのアイコンタクト検出を行い、TempEC-HPでは目画像に加えて顔方向も利用します。<br/>



# パラメータ 
各パラメータは以下のように設定してあります。必要に応じて修正してください。<br/>

LSTMのタイムステップ = 10<br/>
識別の閾値 = 0.5<br/>
LSTM学習のバッチサイズ = 128<br/>
画像サイズ = 36 × 60<br/>
