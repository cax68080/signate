#選択した特徴量を使ってモデリング
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
#データ読み込む
df = pd.read_csv(".\\SIGNATE\\健診データによる肝疾患判定\\train.csv")
#前処理
df["AG_ratio"].fillna(df["Alb"]/(df["TP"]-df["Alb"]),inplace=True)
#Maleが1、Famaleが0になるようにGender列をダミー化
df["Gender"] = df["Gender"].apply(lambda x: 1 if x == 'Male' else 0)
#説明変数のデータフレーム
X = df.drop(['disease'],axis=1)
y = df['disease']
#Gender列を除外(数量変数のデータに絞る)
X_target = X.drop(['Gender'],axis=1)
#多項式・交互作用特徴量の生成
polynomial = PolynomialFeatures(degree=2,include_bias=False)
polynomial_arr = polynomial.fit_transform(X_target)
#polynomial_arrのデータフレーム化
X_polynomial = pd.DataFrame(polynomial_arr,columns=['poly' +str(x) for x in range(polynomial_arr.shape[1]) ])
#組込法のモデル・閾値の設定
fs_model = LogisticRegression(penalty='l1',solver='liblinear',random_state=0)
fs_threshold = 'mean'
#組込法モデルの初期化
selector = SelectFromModel(fs_model,threshold=fs_threshold)
#特徴量選択の実行
selector.fit(X_polynomial,y)
mask = selector.get_support()
#選択された特徴量だけのサンプル取得
X_polynomial_masked = X_polynomial.loc[:,mask]
#学習用・評価用データの分割
X_train,X_test,y_train,y_test = train_test_split(X_polynomial_masked,y,test_size=0.3,random_state=0)
#モデルの学習・予測
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict_proba(X_test)[:,1]
#ROC曲線の描画
fpr,tpr,thresholds = roc_curve(y_true=y_test,y_score=y_pred)
plt.plot(fpr,tpr,label='roc_curve')
plt.plot([0,1],[0,1],linestyle=':',label='random')
plt.plot([0,0,1],[0,1,1],linestyle=':',label='ideal')
plt.legend()
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.show()
#AUCスコアの算出
auc_score = roc_auc_score(y_true=y_test,y_score=y_pred)
print('AUC:',auc_score)
#評価データを読み込む
df_test = pd.read_csv(".\\SIGNATE\\健診データによる肝疾患判定\\test.csv")
#Gender列を除外(数量変数のデータに絞る)
X_test_target = df_test.drop(['Gender'],axis=1)
#多項式・交互作用特徴量の生成
polynomial = PolynomialFeatures(degree=2,include_bias=False)
polynomial_test_arr = polynomial.fit_transform(X_test_target)
#polynomial_arrのデータフレーム化
X_test_polynomial = pd.DataFrame(polynomial_test_arr,columns=['poly' +str(x) for x in range(polynomial_test_arr.shape[1]) ])
print(X_polynomial.shape)
print(X_polynomial.head())
print(X_test_polynomial.shape)
print(X_test_polynomial.head())
#選択された特徴量だけのサンプル取得
X_test_polynomial_masked = X_test_polynomial.loc[:,mask]
#学習用データで作成したモデルで評価用データを投入して予測
y_test_pred = model.predict_proba(X_test_polynomial_masked)[:,1]
print(y_test_pred)
y_test_predict = model.predict(X_test_polynomial_masked)
print(y_test_predict)
df_test["disease"] = y_test_predict
print(df_test.head())
df_test.to_csv(".\\SIGNATE\\健診データによる肝疾患判定\\result.csv")
#fpr,tpr,thresholds = roc_curve(y_true=y,y_score=y_test_pred)
#plt.plot(fpr,tpr,label='roc_curve')
#plt.plot([0,1],[0,1],linestyle=':',label='random')
#plt.plot([0,0,1],[0,1,1],linestyle=':',label='ideal')
#plt.legend()
#plt.xlabel('false positive rate')
#plt.ylabel('true positive rate')
#plt.show()
#AUCスコアの算出
#auc_score = roc_auc_score(y_true=y_test,y_score=y_test_pred)

