#量的データと質的データの可視化
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
#mpgとcylindersの相関関係
sns.boxplot('cylinders','mpg',data=df_train)
plt.show()
#mpgとmodel yearの相関関係
sns.boxplot('model year','mpg',data=df_train)
plt.show()
#mpgとoriginの相関関係
sns.boxplot('origin','mpg',data=df_train)
plt.show()



