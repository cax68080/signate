# ライブラリのインポート
from sklearn.metrics import accuracy_score

# 予測ラベル
y_pred = [0, 2, 1, 3]

# 正解ラベル
y_true = [0, 2, 1,3]

# accuracyの算出
print(accuracy_score(y_true, y_pred))