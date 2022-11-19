#first_review、last_reviewの欠損値を埋める
import pandas as pd
train_data = pd.read_csv('train.csv')
#欠損値の確認
print(train_data.isnull().sum())
#first_reviewの情報を確認する
print(train_data.info)
print(train_data['first_review'].head())
#first_reviewが欠損値になっているデータを抜き出す
first_review_data = train_data[train_data['first_review'].notnull()]
print(first_review_data.value_counts('first_review'))
