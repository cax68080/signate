import pandas as pd
#学習用データを読み込む
df = pd.read_csv('E:\Documents\Python\SIGNATE\株価の推移予測\\train.csv')
#先頭5件を読み込む
print(df.head())
#データ数・カラム数を表示する
print(df.shape)
#データ情報を表示する
print(df.info())
#Dateのデータ型を変換する
print(df['Date'].head())
df['Date'] = pd.to_datetime(df['Date'],format="%Y-%m-%d")
print(df['Date'].head())
print(df.info())
#データを日付順に並び変える
df.sort_values(by='Date',ascending=True,inplace=True)
print(df.head())
print(df.tail())
#インデックスをDateに更新する
df.set_index(keys='Date',inplace=True)
print(df.head())

##株価トレンドの確認
#matplotlibをインポートする
from matplotlib import pyplot as plt

#基本統計量を出力する
print(df.describe())

#量的データの時系列推移
#特定のカラムの抽出
df_new = df.loc[:,["Open","High","Low","Close"]]
df_new.plot(kind='line')
plt.show()

#量的データの時系列推移(短期的な株価の推移)
#ライブラリのインポート
from datetime import datetime
#特定のカラムの抽出
df_new = df.loc[df.index > datetime(2011,1,1),["Open","High","Low","Close"]]
df_new.plot(kind='line')
plt.show()

#質的データの種類と出現数
#データフレームからUpのみを抽出
sr_up = df['Up']
print(sr_up.head())
#Upの出現数の確認
print(sr_up.value_counts())

#株価の上昇回数の描画
sr_up.plot(kind='line')
plt.show()

#データのサンプリング
#シリーズを月ごとにサンプリング
up_monthly = sr_up.resample(rule='M')

#サンプルの平均値の算出
up_mean = up_monthly.mean()

#折れ線グラフの描画
up_mean.plot(kind='line')
plt.show()

#データ前処理
#始値－終値差分値の作成
df['Body'] = df['Open'] - df['Close']
print(df.head())

#学習・評価データの分割
#ライブラリのインポート
from sklearn.model_selection import train_test_split

#目的変数をy_dataに格納する
y_data = df['Up']

#説明変数をx_dataに確認する。
X_data = df.drop(columns=['Up'],inplace=False)

print(y_data.head())
print(X_data.head())

#学習データ・検証データと評価データに80:20に分割する
X_trainval,X_test,y_trainval,y_test = train_test_split(X_data,y_data,test_size=0.20,shuffle=False)

#学習データと検証データに75:25に分割する
X_train,X_val,y_train,y_val = train_test_split(X_trainval,y_trainval,test_size=0.25,shuffle=False)

#分割結果をグラフに表示する。
X_train['Body'].plot(kind='line')
X_val['Body'].plot(kind='line')
X_test['Body'].plot(kind='line')

#折れ線グラフに凡例を表示する
plt.legend(['Train','Val','Test'])
plt.show()

#データ整形
#ライブラリのインポート
from sklearn.preprocessing import StandardScaler
import numpy as np

#5日ごとにデータをまとめる
#関数get_standardized_tを定義する
def get_standardized_t(X,num_date):
    #入力データをNumpy配列に変換
    X = np.array(X)
    X_t_list = []
    for i in range(len(X) - num_date + 1):
        X_t = X[i:i+num_date]
        scaler = StandardScaler()
        X_standardized = scaler.fit_transform(X_t)
        X_t_list.append(X_standardized)
    #Numpy配列をreturn
    return np.array(X_t_list)

#期間の設定
num_date = 5
#学習用、検証用、評価用データの加工
X_train_t = get_standardized_t(X=X_train,num_date=num_date)
X_val_t = get_standardized_t(X=X_val,num_date=num_date)
X_test_t = get_standardized_t(X=X_test,num_date=num_date)
#学習用、検証用、評価用データの表示
print(X_train_t.shape)
print(X_val_t.shape)
print(X_test_t.shape)

#目的変数の変形
y_train_t = y_train[num_date-1 :]
y_val_t = y_val[num_date-1 :]
y_test_t = y_test[num_date-1 :]

#目的変数の形の表示
print(y_train_t.shape)
print(y_val_t.shape)
print(y_test_t.shape)

print(y_train_t.mean)
print(y_val_t.mean)
print(y_test_t.mean)

#kerasのインポート
from keras.models import Sequential
from keras.layers import Dense,LSTM

#LTSMネットワークの定義
#ネットワーク各層のサイズを定義
num_l1 = 100
num_l2 = 20
num_output = 1

#ネットワークを構築
model = Sequential()
#第１層
model.add(LSTM(units=num_l1,activation='tanh',batch_input_shape=(None,X_train_t.shape[1],X_train_t.shape[2])))
#第２層
model.add(Dense(num_l2,activation='relu'))
#出力層
model.add(Dense(num_output,activation='sigmoid'))
#ネットワークのコンパイル
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# モデルの学習の実行（学習の完了までには数秒から数十秒ほど時間がかかります。）
result = model.fit(x=X_train_t, y=y_train_t, epochs=80, batch_size=24, validation_data=(X_val_t, y_val_t))








