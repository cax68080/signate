import pandas as pd

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
print(train_data.isnull().sum())
#cityの種類を確認する
print(train_data['city'].unique())
#accommodatesの種類を確認する
print(train_data['accommodates'].unique())
#cleaning_feeの種類を確認する
print(train_data['cleaning_fee'].unique())
#bed_type
print(train_data['bed_type'].unique())
#bedrooms
print(train_data['bedrooms'].unique())
#beds
print(train_data['beds'].unique())
#property_type
print(train_data['property_type'].unique())
#room_type
print(train_data['room_type'].unique())
#name
print(train_data['name'].unique())
#各列の相関係数を確認する
print(train_data.corr())

#seabornとmatplotlibのインポート
import matplotlib.pyplot as plt
import seaborn as sns
#yのヒストグラム
y_low = train_data[train_data['y'] <= 750]
y_low = y_low['y']
y_low.plot.hist(title='宿泊価格')
plt.show()
y_high = train_data[train_data['y'] > 750]
y_high = y_high['y']
y_high.plot.hist(title='宿泊価格')
plt.show()
#accommodatesとyの箱ひげ図
sns.boxplot(x='accommodates',y='y',data=train_data)
#plt.ylim(0,600)
plt.show()
#cityとyの箱ひげ図
sns.boxplot(x='city',y='y',data=train_data)
plt.show()
#room_typeとyの箱ひげ図
sns.boxplot(x='room_type',y='y',data=train_data)
plt.ylim(0,500)
plt.show()
#bathroomsとyの箱ひげ図
sns.factorplot(x='bathrooms',y='y',kind='box',data=train_data)
#plt.ylim(0,500)
plt.show()
bathrooms_data = train_data[(train_data['bathrooms'] == 0.0) & (train_data['y'] >= 1000)]
print(bathrooms_data.shape)
sns.factorplot(x='name',y='y',kind='box',data=bathrooms_data)
#bed_typeとyの箱ひげ図
sns.factorplot(x='bed_type',y='y',kind='box',data=train_data)
plt.ylim(0,500)
plt.show()
#bedroomsとyの箱ひげ図
sns.factorplot(x='bedrooms',y='y',kind='box',data=train_data)
#plt.ylim(0,500)
plt.show()
#bedsとyの箱ひげ図
sns.factorplot(x='beds',y='y',kind='box',data=train_data)
#plt.ylim(0,500)
plt.show()

