#質的データの頻度値の算出
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
#cylindersの頻度値
cylinders_var = df_train['cylinders']
cylinders_count = cylinders_var.value_counts(sort=False)
cylinders_count.plot.bar(title='cylinders')
plt.xlabel('cylinders')
plt.ylabel('count')
plt.show()
#model yearの頻度値
model_year_var = df_train['model year']
model_year_count = model_year_var.value_counts(sort=False)
model_year_count.plot.bar(title='model year')
plt.xlabel('model year')
plt.ylabel('count')
plt.show()
#originの頻度値
origin_var = df_train['origin']
origin_count = origin_var.value_counts(sort=False)
origin_count.plot.bar(title='origin')
plt.xlabel('origin')
plt.ylabel('count')
plt.show()
