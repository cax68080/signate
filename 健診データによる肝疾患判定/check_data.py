#データの特徴の洗い出し
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(".\\SIGNATE\\健診データによる肝疾患判定\\train.csv")
df["AG_ratio"].fillna(df["Alb"]/(df["TP"]-df["Alb"]),inplace=True)
#print(df.describe())
#データフレームを数量変数とカテゴリ変数に分ける
#カテゴリ変数の列名を抽出する
col_categoric = ['Gender','disease']
#数量変数のデータフレームを作成
df_numeric = df.drop(['id','Gender','disease'],axis=1)
print(df_numeric)
#カテゴリ変数のデータフレームを作成
df_categoric = df[col_categoric]
#print(df_categoric)
#disease列のカテゴリごとの個数を調べる
count_disease = df_categoric['disease'].value_counts()
count_gender = df_categoric['Gender'].value_counts()
#カテゴリ変数の可視化
count_disease.plot(kind='bar')
plt.show()
count_gender.plot(kind='bar')
plt.show()
#数量変数の可視化(ヒストグラム)
df_numeric.hist(figsize=(8,6))
plt.tight_layout()
plt.show()
#df_categoricのdiseaseにdf_numericを結合する
df_tmp = pd.concat([df_categoric['disease'],df_numeric],axis=1)
#print(df_tmp.head())
#disease=0(疾患なし)のサンプルを表示
#print(df_tmp.query('disease==0').head())
#disease=1(疾患あり)のサンプルを表示
#print(df_tmp.query('disease==1').head())
#diseaseの値に応じたAge列の抽出
#df_Age_non = df_tmp.query('disease==0')['Age']
#df_Age_disease = df_tmp.query('disease==1')['Age']
#2つのデータフレームのヒストグラムを作成
#sns.distplot(df_Age_non)
#sns.distplot(df_Age_disease)
#凡例の表示
#plt.legend(labels=['non','diseased'],loc='upper right')
#plt.show()
#diseaseの値に応じたT_Bil列の抽出
#df_TBil_non = df_tmp.query('disease==0')['T_Bil']
#df_TBil_disease = df_tmp.query('disease==1')['T_Bil']
#2つのデータフレームのヒストグラムを作成
#sns.distplot(df_TBil_non)
#sns.distplot(df_TBil_disease)
#凡例の表示
#plt.legend(labels=['non','diseased'],loc='upper right')
#plt.xlim(0.0,5.0)
#plt.show()
#グラフの表示
plt.figure(figsize=(12,12))

for ncol,colname in enumerate(df_numeric.columns):
    print(ncol)
    plt.subplot(3,3,ncol+1)
    sns.distplot(df_tmp.query('disease==0')[colname])
    sns.distplot(df_tmp.query('disease==1')[colname])
    plt.legend(labels=['non','disease'],loc='upper right')
plt.show()
#print(df[["T_Bil","D_Bil"]].head())
#print(df[["T_Bil","D_Bil"]].corr())
#print(df[["T_Bil","Age"]].head())
#print(df[["T_Bil","Age"]].corr())
#heatmapを表示する
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),vmin=1.0,vmax=1.0,annot=True,cmap="coolwarm",linewidths=0.1)
plt.show()



