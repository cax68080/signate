#相関係数の算出
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
#相関係数を求める
print(df_train.corr())
#相関係数の可視化
corr_matrix = df_train.corr()
sns.heatmap(corr_matrix)
plt.show()
