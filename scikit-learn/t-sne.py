from sklearn.datasets import load_iris,load_wine
from sklearn.manifold import TSNE

#データの準備
iris = load_iris()
X = iris.data
y = iris.target

#次元削減前
print(X)
print(X.shape)

#インスタンス作成
tsne = TSNE(n_components=2,random_state=0)

#次元削減後
print(tsne.fit_transform(X))
print(tsne.fit_transform(X).shape)