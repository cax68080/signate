import pandas as pd
train_data = pd.read_csv('train.csv')
#欠損値の確認
print(train_data.isnull().sum())
#zipcodeが欠損値になっているデータを抜き出す
bedrooms_nan_data = train_data[train_data['zipcode'].isnull()]
print(bedrooms_nan_data['zipcode'])
print(train_data['zipcode'].head())