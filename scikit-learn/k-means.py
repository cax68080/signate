from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# データの準備
iris = load_iris()
X = iris.data
y = iris.target

# インスタンス作成
kms = KMeans(n_clusters=3,random_state=0)

# クラスタリング実行
print(kms.fit_predict(X))