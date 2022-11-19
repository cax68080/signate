#走行距離予測
from numpy.core.numeric import NaN
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_log_error as MSE
from matplotlib import pyplot as plt
df_train = pd.read_csv('E:\Documents\Python\SIGNATE\自動車の走行距離予測\\train.tsv',sep='\t')
df_test = pd.read_csv('E:\Documents\Python\SIGNATE\自動車の走行距離予測\\test.tsv',sep='\t')
df_sample = pd.read_csv('E:\Documents\Python\SIGNATE\自動車の走行距離予測\sample_submit.csv')
df_train['horsepower'] = df_train.replace('?',np.nan)
df_train['horsepower'] = df_train['horsepower'].fillna(df_train['horsepower'].mean())
#df_train.info()
df_test['horsepower'] = df_test.replace('?',np.nan)
df_test['horsepower'] = df_test['horsepower'].fillna(df_test['horsepower'].mean())
#df_test.info()
#目的変数yを作成する
y_train = df_train['mpg']
#説明変数xを作成する
x_train = df_train[['cylinders','displacement','weight','model year','origin']]
x_test =  df_test[['cylinders','displacement','weight','model year','origin']]
#学習データと評価データに分割する
#x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
#print(x_train.shape)
#print(x_test.shape)
#モデルの箱を準備する
lr =LR()
#モデルを学習する
lr.fit(x_train,y_train)
#モデルから予測値を求める
y_pred_train = lr.predict(x_train)
y_pred_test = lr.predict(x_test)
#print(y_pred_train.size)
#print(y_pred_train)
#print(y_pred_test)
#RMSEを算出する
mse_train = MSE(y_train,y_pred_train)
mse_test = MSE(y_train,y_pred_test)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)
print(rmse_train)
print(rmse_test)
#散布図を作成する
plt.figure(figsize=(5,5))
plt.scatter(y_train,y_pred_train)
plt.show()
plt.figure(figsize=(5,5))
plt.scatter(y_train,y_pred_test)
plt.show()
#ファイル作成
df_test['mpg'] = y_pred_test
print(df_test.info)
df_test[['id','mpg']].to_csv('E:\Documents\Python\SIGNATE\自動車の走行距離予測\submit.csv',header=False,index=False)