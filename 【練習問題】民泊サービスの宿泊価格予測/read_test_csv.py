import pandas as pd
test_data = pd.read_csv('test.csv')
#データ数の確認
print(test_data.shape)
#データ情報
print(test_data.info())
#test_dataの先頭５行を表示
print(test_data.head())
#test_dataの列を確認
print(test_data.columns)
#欠損値の確認
print(test_data.isnull().sum())
#bathroomsの欠損値を平均値の1で穴埋めする
print(test_data.fillna({'bathrooms':1},inplace=True))
#bedroomsの欠損値を平均値の1で穴埋めする