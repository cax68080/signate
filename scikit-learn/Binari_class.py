#ライブラリのインポート
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  

#データロード
breast = load_breast_cancer()
df_breast = pd.DataFrame(data=breast.data,columns=breast.feature_names)
df_breast['target'] = breast.target

#インスタンス作成
clf = LogisticRegression(solver='liblinear',C=1000)

#説明変数
X = df_breast[breast.feature_names].values

#目的変数target
Y = df_breast['target'].values

#評価データと検証データに分割する
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#予測モデルを作成する
clf.fit(X_train,Y_train)

#予測したモデルで未知データに対して予測
clf.predict(X_test)
#作成したモデルで未知データの予測
print(clf.predict_proba(X_test))

#精度算出
print(accuracy_score(Y_test,clf.predict(X_test)))