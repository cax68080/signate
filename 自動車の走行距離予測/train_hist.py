#自動車データの特徴把握
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
#学習用データを読み込む
df_train = pd.read_csv('E:\Documents\Python\SIGNATE\自動車の走行距離予測\\train.tsv',sep='\t')
#df_test = pd.read_csv('E:\Documents\Python\SIGNATE\自動車の走行距離予測\\test.tsv',sep='\t')
#異常値の補正
df_train['horsepower'] = df_train.replace('?',np.nan)
df_train['horsepower'] = df_train['horsepower'].fillna(df_train['horsepower'].mean())
#df_train.info()
#不要データを削除する
df_train = df_train.drop(['id','car name'],axis=1)
#df_train.info()
#print(df_train.describe())
#print(type(df_train))
mpg_var = df_train['mpg']
mpg_var.plot.hist(title='mpg')
plt.show()
displacement_var = df_train['displacement']
displacement_var.plot.hist(title='displacement')
plt.show()
weight_var = df_train['weight']
weight_var.plot.hist(title='weight')
plt.show()
horsepower_var = df_train['horsepower']
horsepower_var.plot.hist(title='horsepower')
plt.show()
