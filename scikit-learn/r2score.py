# ライブラリのインポート
from sklearn.metrics import r2_score

# 正解値
y_true = [3, -0.5, 2, 7]

# 予測値
y_pred = [2.5, 0.5, 2, 8.5]

# accuracyの算出
print(r2_score(y_true, y_pred))