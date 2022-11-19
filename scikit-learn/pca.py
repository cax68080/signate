from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# データの準備
iris = load_iris()
X = iris.data
y = iris.target

# 次元削減前
print(X)

# インスタンス作成からPCA実行
pca = PCA(n_components=1)
pca.fit(X)
X_ = pca.transform(X)

# 次元削減後
print(X_)