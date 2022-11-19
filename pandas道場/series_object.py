import numpy as np
import pandas as pd

ser1 = pd.Series([0,1,1,2,2,3,np.nan])
print(ser1)
print(len(ser1))
print(ser1.shape)
print(ser1.count())
print(ser1.nunique())
print(len(ser1.value_counts()))

ser2 = pd.Series(['apple', 'apricot', 'avocado',
                  'banana', 'blueberry'])
print(ser2.head(3))
print(ser2.tail(2))

ser3 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
ser4 = pd.Series([1, 'b', 3], index=['a', 2, 'c'])
ser5 = pd.Series(['a', 'b', 'c'], index=[1, 2, 3])
print(ser3)
print(ser4)
print(ser5)
print(ser3.loc[['a']])
#print(ser3.iloc[0,1])
print(ser4.loc[['a',2]])
print(ser5.loc[3])

a1 = np.array([0,1,2,np.nan])
ser6 = pd.Series(a1)
print(ser6)
print(a1.mean())
print(ser1.mean())
print(ser1.mean(skipna=False))

a2 = np.arange(10)
ser7 = pd.Series(a2)
print(a2)
print(ser7)
print(ser7[[True] * 10])
print(ser7[a2 >= 10])
print(ser7[(ser7 > 2)&(a2 < 5)])
#print(ser7[(ser7 == 0) and (ser7 < 2)])
#print(ser7[(ser7 > 0 | a2 < 7)])

ser8 = pd.Series(['k', 'K', 'M', 'O'],  
                 index=['kiwi', 'kumquat',  
                        'mango', 'orange'])
ser8['kiwi'] = 'K'
print(ser8)
ser8['lime'] = 'L'
print(ser8)
del ser8['mango']
print(ser8)

ser9 = pd.Series([1, 2, 3, 4], index=[4, 3, 2, 1])
ser10 = ser9.copy()
ser11 = ser10
ser10.iloc[0] = 3
ser12 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
print(ser9[3])
print(ser10[4])
print(ser11[2])
print(len(ser12[:'b']))