import pandas as pd
from sklearn.model_selection import train_test_split

#train.csv を読み込む
train_data = pd.read_csv('train.csv')
#列を表示
print(train_data.columns)
#データ数を確認
print(train_data.shape)
#データ情報を確認
print(train_data.info())
#欠損値を確認
print(train_data.isnull().sum())
#欠損値の処理
#bathroomsの欠損値を平均値の1で埋める
train_data['bathrooms'].fillna(1,inplace=True)
#bedroomsの欠損値を平均値の1で埋める
train_data['bedrooms'].fillna(1,inplace=True)
#bedsの欠損値を平均値の1で埋める
train_data['beds'].fillna(1,inplace=True)
#first_review,last_reviewの欠損値を"2017-01-01"で埋める
train_data.fillna({'first_review':'2017-01-01','last_review':'2017-01-01'},inplace=True)
#host_response_rateを"0%"で埋める
train_data['host_response_rate'].fillna('0%',inplace=True)
#host_has_profile_picが欠損しているデータを削除する
train_data.dropna(subset=['host_has_profile_pic'],inplace=True)
# review_scores_ratingを96.0で埋める
train_data['review_scores_rating'].fillna(96.0,inplace=True)
#zipcodeを'00000'で埋める
train_data['zipcode'].fillna('00000',inplace=True)
#欠損値を確認
#print(train_data.isnull().sum())
#取り出す列を選択する
select_col = ['room_type','accommodates','bathrooms','bed_type','bedrooms','city','cleaning_fee']
train_data_edt = train_data[select_col]
#print(train_data_edt.columns)
#print(train_data_edt.head())
#ダミー変数化
dummy_train_data_edt = pd.get_dummies(train_data_edt,drop_first=True)
print(dummy_train_data_edt.columns)
#print(dummy_train_data_edt.head())
#ダミーデータとyを評価用と検証用データに分ける
X_train,X_test,y_train,y_test = train_test_split(dummy_train_data_edt, train_data['y'], random_state = 1234)
print(X_train.shape)
print(X_test.shape)

#モデルの作成
from sklearn.linear_model import LinearRegression

#モデルの準備
lr = LinearRegression()

#モデルの学習
lr.fit(X_train,y_train)

#予測
y_pred_train = lr.predict(X_train)
#print(y_pred_train)

#予測精度の評価
from sklearn.metrics import  mean_squared_error as MSE
import numpy as np

#RMSEを算出
rmse_train = np.sqrt(MSE(y_train,y_pred_train))
#print(rmse_train)

#X_testで予測値を算出する
y_pred_test = lr.predict(X_test)

#予測精度の評価
rmse_test = np.sqrt(MSE(y_test,y_pred_test))
#print(rmse_test)

#test.csvを読み込む
test_data = pd.read_csv('test.csv')
#print(test_data.isnull().sum())
#bathroomsの欠損値を平均値の1で埋める
test_data['bathrooms'].fillna(1,inplace=True)
#bedroomsの欠損値を平均値の1で埋める
test_data['bedrooms'].fillna(1,inplace=True)
#評価対象の列を取り出す
test_data_edt = test_data[select_col]
#print(test_data_edt.columns)
#ダミー変数化
dummy_test_data_edt = pd.get_dummies(test_data_edt,drop_first=True)
#print(dummy_test_data_edt.isnull().sum())
#print(dummy_test_data_edt.columns)
y_pred_test = lr.predict(dummy_test_data_edt)
y_pred_result = np.array([])
for i in y_pred_test:
    y_pred_result = np.append(y_pred_result,round(i))

df_y_pred_result = pd.DataFrame(y_pred_result)
df_y_pred_result.to_csv('result.csv')
