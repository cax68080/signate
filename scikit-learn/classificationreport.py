# ライブラリのインポート
from sklearn.metrics import classification_report

# 正解ラベル
y_true = [0, 1, 2, 2, 2]

# 推定ラベル
y_pred = [0, 1, 2, 2, 2]

# 目的変数の各クラス名
target_names = ['class 0', 'class 1', 'class 2']

print(classification_report(y_true, y_pred, target_names=target_names))