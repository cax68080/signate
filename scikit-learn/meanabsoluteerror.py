# ライブラリのインポート
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# データのロード
boston = load_boston()
df_boston = pd.DataFrame(data=boston.data,columns=boston.feature_names)
df_boston['target'] = boston.target

# インスタンス作成
clf = KNeighborsRegressor()

# 説明変数
X = df_boston[boston.feature_names].values

# 目的変数target
Y = df_boston['target'].values

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.5,random_state=0)

# 予測モデルを作成
clf.fit(X_train, y_train)

# MAE算出
print(mean_absolute_error(y_test, clf.predict(X_test)))