import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

# データのロード
breast = load_breast_cancer()
df_breast = pd.DataFrame(data=breast.data,columns=breast.feature_names)
df_breast['target'] = breast.target

#インスタンス作成
clf = RandomForestClassifier(random_state=0)

#説明変数
X = df_breast[breast.feature_names].values

#目的変数
y = df_breast['target'].values

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=0)

# 予測モデルを作成
clf.fit(X_train, y_train)

# 推定ラベル
y_pred = clf.predict(X_test)

# 混合行列の作成
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ヒートマップの作成
sns.heatmap(cm, annot=True,cmap='Blues')
plt.show()