import pandas as pd
train_data = pd.read_csv('train.csv')
#欠損値の確認
print(train_data.isnull().sum())
#host_response_rateが欠損値になっているデータを抜き出す
bedrooms_nan_data = train_data[train_data['host_response_rate'].isnull()]
print(bedrooms_nan_data['host_response_rate'])
print(train_data['host_response_rate'].value_counts())