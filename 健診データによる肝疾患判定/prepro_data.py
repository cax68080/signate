#データの特徴の洗い出し
from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
#データ読み込む
df = pd.read_csv(".\\SIGNATE\\健診データによる肝疾患判定\\train.csv")
#print(df.head())
#前処理
df["AG_ratio"].fillna(df["Alb"]/(df["TP"]-df["Alb"]),inplace=True)
#print(df.head())
#Maleが1、Famaleが0になるようにGender列をダミー化
df["Gender"] = df["Gender"].apply(lambda x: 1 if x == 'Male' else 0)
#print(df.head())
#説明変数のデータフレーム
X = df.drop(['disease'],axis=1)
#目的変数のデータフレーム
y = df['disease']
#print(X)
#print(y)
#学習用データと評価用データに分割する
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
print(X_train.shape)
print(y_train.shape)
