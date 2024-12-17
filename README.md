# PBL05_ex03_app_MDX2024

MDX2024 PBL05 演習03 評価値/残差 計算アプリ

## 事前準備

アプリで使用しているライブラリのインストール

``` shell
pip install -r ./requirements.txt
```

## アプリ起動方法

stramlitにアプリファイルを指定して起動

``` shell
python -m streamlit run ./streamlit_app.py
```

開発中は以下のようにアプリファイル更新されたら再度実行オプションを付加すると楽

``` shell
python -m streamlit run --server.runOnSave True ./streamlit_app.py
```

## 備考

Streamlit Commnity Cloud 公開するつもりで作った。  
Competitionサイトで公開されたデータを世間一般に公開するのが問題になるため、  
使うたびにユーザーに

* 作業実績データファイル(actual_test.csv)
* 正解データファイル(PBL05((工数予測))_演習03_解答.csv)

をアップロードしてもらうように作った。

ローカルでアプリを立ち上げる場合、ルートフォルダに上記2ファイルを置いておけば  
毎度アップロードしなくてもいいように分岐処理を入れている
