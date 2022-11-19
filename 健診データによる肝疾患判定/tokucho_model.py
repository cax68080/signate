#検査データの特徴を利用したモデル改善
from numpy.core.fromnumeric import shape
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.preprocessing import PolynomialFeatures
#データ読み込む
df = pd.read_csv(".\\SIGNATE\\健診データによる肝疾患判定\\train.csv")
#前処理
df["AG_ratio"].fillna(df["Alb"]/(df["TP"]-df["Alb"]),inplace=True)
#Maleが1、Famaleが0になるようにGender列をダミー化
df["Gender"] = df["Gender"].apply(lambda x: 1 if x == 'Male' else 0)
#説明変数のデータフレーム
X = df.drop(['disease'],axis=1)
y = df['disease']
#等間隔のbin分割
#X_cut,bin_indice =pd.cut(X['T_Bil'],bins=100,retbins=True)
#bin分割した結果の表示
#print("binの区切り：",bin_indice)
#print("--- bin区切りごとのデータ数 ---")
#print(X_cut.value_counts())
#境界値を指定したbinの分割
bins_T_Bil = [0,0.5,1.0,100]
X_cut,bin_indice = pd.cut(X["T_Bil"],bins=bins_T_Bil,retbins=True,labels=False)
#bin分割した結果をダミー変数化
X_dummies =pd.get_dummies(X_cut,prefix=X_cut.name)
#元の説明変数のデータフレーム(x)と、ダミー変数化の結果(x_dummies)を横連結
X_binned = pd.concat([X,X_dummies],axis=1)
#学習用・評価用データの分割
X_train,X_test,y_train,y_test = train_test_split(X_binned,y,test_size=0.3,random_state=0)
#モデルの学習・予測
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict_proba(X_test)[:,1]

#ROC曲線の描画(偽陽性率・真陽性率・閾値の算出)
fpr,tpr,thresholds = roc_curve(y_true=y_test,y_score=y_pred)
plt.plot(fpr,tpr,label='roc_curve')
plt.plot([0,1],[0,1],linestyle=':',label='random')
plt.plot([0,0,1],[0,1,1],linestyle=':',label='ideal')
plt.legend()
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.show()
auc_score = roc_auc_score(y_true=y_test,y_score=y_pred)
print('AUC:',auc_score)

#Gender列を除外(数量変数のデータに絞る)
X_target = X.drop(['Gender'],axis=1)
#多項式・交互作用特徴量の生成
polynomial = PolynomialFeatures(degree=2,include_bias=False)
polynomial_arr = polynomial.fit_transform(X_target)

#polynomial_arrのデータフレーム化
X_polynomial = pd.DataFrame(polynomial_arr,columns=['poly' +str(x) for x in range(polynomial_arr.shape[1]) ])

#生成した多項式・交互作用特徴量の表示
print(X_polynomial.shape)
print(X_polynomial.head())