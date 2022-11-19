#データの読み込みと整形
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(".\\SIGNATE\\健診データによる肝疾患判定\\train.csv")
#print(df.shape)
#print(df.info())
#print(df.isnull().head())
#print(df.isnull().sum())
#print(df[df.isnull().any(axis=1)])
print(df.loc[[31,278,495,648],:])
df["AG_ratio"].fillna(df["Alb"]/(df["TP"]-df["Alb"]),inplace=True)
print(df.loc[[31,278,495,648],:])
#print(df.duplicated().sum())