import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

iris = load_iris()
df_iris = pd.DataFrame(data=iris.data,columns=iris.feature_names)
df_iris['target'] = iris.target
#print(df_iris.head())
#print(df_iris.info())
#print(df_iris.describe())
#df_iris.to_csv('E:\Documents\Python\SIGNATE\scikit-learn\iris.csv')
#データの標準化
sc = StandardScaler()
sc.fit(df_iris)
df_iris_sc = pd.DataFrame(sc.transform(df_iris), columns=df_iris.columns)
#print(df_iris_sc.head())
#print(df_iris_sc.describe()['sepal length (cm)'])
#データの正規化
ms = MinMaxScaler([0,1])
ms.fit(df_iris)
df_iris_ms = pd.DataFrame(ms.transform(df_iris),columns=df_iris.columns)
#print(df_iris_ms.describe().loc[['min','max']])
#単回帰分析
#インスタンス作成
clf = LinearRegression()
#説明変数
X = df_iris['sepal length (cm)'].values.reshape(-1,1)
#目的変数
Y = df_iris['target'].values
#予測モデルを作成
clf.fit(X,Y)
#回帰係数
print(clf.coef_)
#切片
print(clf.intercept_)
#重回帰分析
#インスタンス作成
clf = LinearRegression(normalize=True)
#説明変数
X = df_iris[iris.feature_names].values
#目的変数
Y = df_iris['target'].values
#予測モデルを作成
clf.fit(X,Y)
#回帰係数
print(clf.coef_)
#切片
print(clf.intercept_)
