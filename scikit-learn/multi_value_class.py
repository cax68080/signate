#ライブラリのインポート
import pandas as pd
from sklearn.datasets import load_wine,load_boston
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  

#データのロード
wine = load_wine()
df_wine = pd.DataFrame(data=wine.data,columns=wine.feature_names)
df_wine['target'] = wine.target

#インスタンス作成
clf = LogisticRegression(solver='liblinear')

#説明変数
X = df_wine[wine.feature_names].values

#目的変数
Y = df_wine['target'].values

#データの分割
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

#予測モデルを作成
clf.fit(X_train,y_train)

#精度を算出
print(accuracy_score(y_test,clf.predict(X_test)))