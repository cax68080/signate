#燃費モデルを作成する
from scipy.sparse.construct import random
import read_data as rd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as MSE
#print(rd.df_train.head())
#mpgを変数yに代入する
y = rd.df_train['mpg']
#print(y.head())
#変数Xに代入する
X = rd.df_train[['cylinders','displacement','horsepower','weight','model year','origin']]
#print(X.head())
#評価用データと検証データに分ける
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
#print(X_train.shape)
#print(X_test.shape)
#モデルの準備と学習
lr = LR()
lr.fit(X_train,y_train)
#モデルから予測結果を求める
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)
#予測結果の評価
#MSEを算出する
mse_train = MSE(y_train,y_pred_train)
rmse_train = rd.np.sqrt(mse_train)
mse_test = MSE(y_test,y_pred_test)
rmse_test = rd.np.sqrt(mse_test)
print(rmse_train)
print(rmse_test)
#予測精度の可視化
rd.plt.figure(figsize=(5,5))
rd.plt.scatter(y_train,y_pred_train)
rd.plt.show()
rd.plt.scatter(y_test,y_pred_test)
#rd.plt.show()
#最小値・最大値を求める
test_min = rd.np.min(y_test)
test_max = rd.np.max(y_test)
print(test_min,test_max)
pred_min = rd.np.min(y_pred_test)
pred_max = rd.np.max(y_pred_test)
print(pred_min,pred_max)
min_value = rd.np.minimum(test_min,pred_min)
max_value = rd.np.maximum(test_max,pred_max)
rd.plt.xlim([min_value,max_value])
rd.plt.ylim([min_value,max_value])
rd.plt.plot([min_value,max_value],[min_value,max_value])
rd.plt.xlabel('実績値')
rd.plt.ylabel('予測値')
rd.plt.show()
