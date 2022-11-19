# ライブラリのインポート
import pandas as pd

# データの読み込み
df_data = pd.read_csv('E:\Documents\Python\SIGNATE\データ前処理道場\pop202003.csv')

#関数設定
def get_ja_name(x):
    ja_name = ''
    for num,i in enumerate(int(x)):
        if(i.isupper()):
            ja_name = x[0:num]
            break;
    return ja_name

def get_en_name(x):
    en_name = ''
    for num,i in enumerate(int(x)):
        if(i.isupper()):
            en_name = x[num:]
            break;
    return en_name


# 日本語の取得
ja_name_list = df_data['市町村名'].map(get_ja_name)

# 英字の取得
en_name_list = df_data['市町村名'].map(get_en_name)

# pandasに代入
df_data['市町村名'] = ja_name_list

df_data['市町村名_読み'] = en_name_list

# 現状のデータをcsv形式にて保存
df_data.to_csv(path_or_buf='E:\Documents\Python\SIGNATE\データ前処理道場\pop202003_1.csv',index=False)







