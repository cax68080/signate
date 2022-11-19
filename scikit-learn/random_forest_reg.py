# ライブラリのインポート
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# データのロード
boston = load_boston()
df_boston = pd.DataFrame(data=boston.data,columns=boston.feature_names)
df_boston['target'] = boston.target

# インスタンス作成
clf = RandomForestRegressor(random_state=0)

# 説明変数
X = df_boston[boston.feature_names].values

# 目的変数target
Y = df_boston['target'].values

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3,random_state=0)

# 予測モデルを作成
clf.fit(X_train, y_train)

# 特徴量重要度
feature_importance = pd.DataFrame({'feature':boston.feature_names,'importances':clf.feature_importances_}).sort_values(by="importances", ascending=False)

# 可視化
sns.barplot(x="importances", y="feature", data=feature_importance.head())
plt.show()