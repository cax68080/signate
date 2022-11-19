#自動車環境性能の改善(評価用データ)
#データ読み込み
#pndasのインポート
import pandas as pd
#numpyのインポート
import numpy as np
#matplotlibのインポート
from matplotlib import pyplot as plt
#モデル作成モジュールインポート
import train_model_dummy as tmd
#評価データの読み込み
df_test = pd.read_csv('E:\Documents\Python\SIGNATE\自動車の走行距離予測\\20220101\\test.tsv',sep='\t')
#評価データの確認
#評価データの異常値をNaNに修正
df_test = df_test.replace('?',np.NaN)
#print(df_test.shape)
#評価データのhorsepowerをfloat64に型変換
df_test['horsepower'] = df_test['horsepower'].astype(np.float64)
#評価データの欠損値を平均値で穴埋め
df_test['horsepower'].fillna(df_test['horsepower'].mean(),inplace=True)
#print(df_test.head())
#評価データの不要なデータを削除する
df_test = df_test.drop(['id','car name','acceleration'],axis=1)
#評価データのcylinders,model year,originを文字列に型変換
df_test[['cylinders','model year','origin']] = df_test[['cylinders','model year','origin']].astype(np.str)
X = pd.get_dummies(df_test)
#print(X.shape)
#モデルから予測結果を求める
y_pred_test = tmd.lr.predict(X)
#print(y_pred_test)
X['mpg'] = y_pred_test
print(X.shape)
X.to_csv('E:\Documents\Python\SIGNATE\自動車の走行距離予測\\20220101\\submit.csv')