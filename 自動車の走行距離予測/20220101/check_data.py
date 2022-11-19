#データ分析
import read_data as rd
import seaborn as sns
#df_trainからcar nameを削除する
df_train = rd.df_train.drop(['car name'],axis=1)
#量的データのヒストグラム
#ヒストグラムを作成する
df_train.hist(figsize=(8,6))
rd.plt.tight_layout()
rd.plt.show()
#質的データ
#シリンダ
cylinders_var = rd.df_train['cylinders']
cylinders_var_count = cylinders_var.value_counts()
cylinders_var_count.plot.bar(title='cylinder')
rd.plt.show()
#年式
model_year_var = rd.df_train['model year']
model_year_var_count = model_year_var.value_counts()
model_year_var_count.plot.bar(title='model year')
rd.plt.show()
#車種
print(len(rd.df_train['car name'].unique()))
#car_name_var = rd.df_train['car name']
#car_name_var_count = car_name_var.value_counts()
#car_name_var_count.plot.barh(title='car name')
#rd.plt.show()
#起源
origin_var = rd.df_train['origin']
origin_var_count = origin_var.value_counts()
origin_var_count.plot.bar(title='origin')
rd.plt.show()
#相関係数
df_train_corr = df_train.corr()
#print(df_train_corr)
#ヒートマップを表示する
sns.heatmap(df_train_corr)
rd.plt.show()
#量的データ同士の相関関係の可視化
#mpgとWeight
rd.plt.scatter(df_train['mpg'],df_train['weight'])
rd.plt.ylabel('weight')
rd.plt.xlabel('mpg')
rd.plt.show()
#mpgとdisplacement
rd.plt.scatter(df_train['mpg'],df_train['displacement'])
rd.plt.ylabel('displacement')
rd.plt.xlabel('mpg')
rd.plt.show()
#mpgとhorsepower
rd.plt.scatter(df_train['mpg'],df_train['horsepower'])
rd.plt.ylabel('horsepower')
rd.plt.xlabel('mpg')
rd.plt.show()
#mpgとcylinders
#rd.plt.scatter(df_train['mpg'],df_train['cylinders'])
#rd.plt.ylabel('cylinders')
#rd.plt.xlabel('mpg')
#rd.plt.show()
#mpgと質的データの相関
#mpgとcylinders
sns.boxplot('cylinders','mpg',data=rd.df_train)
rd.plt.show()
#mpgとmodel year
sns.boxplot('model year','mpg',data=rd.df_train)
rd.plt.show()
#mpgとorigin
sns.boxplot('origin','mpg',data=rd.df_train)
rd.plt.show()
#mpgとcar name
sns.boxplot('car name','mpg',data=rd.df_train)
rd.plt.show()




