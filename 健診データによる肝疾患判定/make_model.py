#データの特徴の洗い出し
#from matplotlib.lines import lineStyles
#from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve
#データ読み込む
df = pd.read_csv(".\\SIGNATE\\健診データによる肝疾患判定\\train.csv")
#前処理
df["AG_ratio"].fillna(df["Alb"]/(df["TP"]-df["Alb"]),inplace=True)
#Maleが1、Famaleが0になるようにGender列をダミー化
df["Gender"] = df["Gender"].apply(lambda x: 1 if x == 'Male' else 0)
#説明変数のデータフレーム
X = df.drop(['disease'],axis=1)
#目的変数のデータフレーム
y = df['disease']
#学習用データと評価用データに分割する
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
#モデルの初期化
lr = LogisticRegression()
#モデルの学習
lr.fit(X_train,y_train)
#予測
y_pred = lr.predict(X_test)
#print(y_pred)
#print(y_pred.shape)
#print(sum(y_pred))
#判定確率の算出
#result = lr.predict_proba(X_test)
#print(result[:5])
#疾患あり(=1)となる確率
#result_1 = lr.predict_proba(X_test)[:,1]
#print(result_1)
#混同行列の作成
cm = confusion_matrix(y_true=y_test,y_pred=y_pred)
#混同行列をデータフレーム化
df_cm = pd.DataFrame(np.rot90(cm,2),index=["actual_Positive","actual_Negative"],columns=["predict_Positive","predict_Negatie"])
print(df_cm)
#heatmapによる混同行列の可視化
sns.heatmap(df_cm,annot=True,fmt="2g",cmap='Blues')
plt.yticks(va='center')
plt.show()
#モデルの予測(疾患あり(=1)に属する確率の算出)
y_pred_prob = lr.predict_proba(X_test)[:,1]
#AUCスコアの算出
auc_score = roc_auc_score(y_true=y_test,y_score=y_pred_prob)
print(auc_score)
#ROC曲線の要素(偽陽性率、真陽性率、閾値)の算出
fpr,tpr,thresholds = roc_curve(y_true=y_test,y_score=y_pred_prob)
#ROC曲線の描画
plt.plot(fpr,tpr,label='roc curve (area = %0.3f)' % auc_score)
plt.plot([0,1],[0,1],lineStyle=':',label='random')
plt.plot([0,0,1],[0,1,1],linestyle=':',label='ideal')
plt.legend()
plt.xlabel('false positive rate')
plt.ylabel('true positive rate' )
plt.show()