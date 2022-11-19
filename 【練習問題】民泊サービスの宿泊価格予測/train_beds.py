import pandas as pd
train_data = pd.read_csv('train.csv')
#欠損値の確認
print(train_data.isnull().sum())
#bedsが欠損値になっているデータを抜き出す
bedrooms_nan_data = train_data[train_data['beds'].isnull()]
print(bedrooms_nan_data['beds'])
print(train_data['bedrooms'].mean())