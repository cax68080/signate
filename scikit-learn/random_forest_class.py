# ライブラリのインポート
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# データのロード
breast = load_breast_cancer()
df_breast = pd.DataFrame(data=breast.data,columns=breast.feature_names)
df_breast['target'] = breast.target

# インスタンス作成
clf = RandomForestClassifier(random_state=0)

# 説明変数
X = df_breast[breast.feature_names].values

# 目的変数target
Y = df_breast['target'].values

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3,random_state=0)

# 予測モデルを作成
clf.fit(X_train, y_train)

# 特徴量重要度
feature_importance = pd.DataFrame({'feature':breast.feature_names,'importances':clf.feature_importances_}).sort_values(by="importances", ascending=False)

print(feature_importance.head())
