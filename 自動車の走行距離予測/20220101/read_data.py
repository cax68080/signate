#自動車環境性能の改善
#データ読み込み
#pndasのインポート
import pandas as pd
#numpyのインポート
import numpy as np
#matplotlibのインポート
from matplotlib import pyplot as plt
#評価データの読み込み
df_train = pd.read_csv('E:\Documents\Python\SIGNATE\自動車の走行距離予測\\20220101\\train.tsv',sep='\t')
#評価データの確認
#評価データの異常値をNaNに修正
df_train = df_train.replace('?',np.NaN)
#評価データの欠損値を削除
df_train = df_train.dropna()
#評価データのhorsepowerをfloat64に型変換
df_train['horsepower'] = df_train['horsepower'].astype(np.float64)
#print(df_train.info())
#評価データの不要なデータを削除する
df_train = df_train.drop(['id'],axis=1)
#print(df_train.head())
#列car nameをダミー変数化
df_dummy = pd.get_dummies(df_train['car name'])
#print(df_dummy.count())
#ダミー変数化したcar nameをデータフレームに結合する
df_train_dummy = df_train.join(df_dummy)
#列car nameを削除する
df_train_dummy = df_train_dummy.drop(['car name'],axis=1)
#print(df_train_dummy.head())

