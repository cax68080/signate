#エルボー法を使ったk-meansクラスタリング
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris,load_wine

# データセットの作成
iris = load_wine()
X = iris.data
Y = iris.target

# エルボー法の実施
distortions = []

# クラスター数1~10を一気に計算
for i in range(1, 14):
    km = KMeans(n_clusters=i)
    # クラスタリングの実行
    km.fit(X)
    # 各々のクラスタでの誤差を算出し、リストに格納
    distortions.append(km.inertia_)
print(distortions)

# グラフのプロット
plt.plot(range(1, 14), distortions, marker="D")
plt.xticks(np.arange(1, 14, 1))
plt.xlabel("Number_of_clusters")
plt.ylabel("Distortion")
plt.show()