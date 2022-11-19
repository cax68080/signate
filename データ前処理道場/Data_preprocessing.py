import pandas as pd
df_data = pd.read_excel(io='E:\Documents\Python\SIGNATE\データ前処理道場\pop202003.xls',sheet_name='市町村別人口')
print(df_data.columns)
df_data.columns = ['市町村名', '2020-03-01_人口', '2020-02-01_人口', '2020-02-01_増減数','2020-02-01-増減率', '2019-02-01_人口', '2019-02-01_増減数', '2019-02-01_増減率']
print(df_data.columns)
df_data = df_data.drop([0,1,2,3,4])
df_data.to_csv(path_or_buf='E:\Documents\Python\SIGNATE\データ前処理道場\pop202003.csv',index=False)