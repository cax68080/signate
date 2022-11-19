import pandas as pd
from numpy.core.numeric import NaN
df_train = pd.read_csv('E:\Documents\Python\SIGNATE\自動車の走行距離予測\\train.tsv',sep='\t')
df_train = df_train.drop(columns=['id'])
df_train = df_train.replace('?',NaN)
df_train = df_train.dropna()
dummy_df = pd.get_dummies(df_train['car name'])
print(dummy_df.head(5))