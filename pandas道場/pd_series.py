import pandas as pd
import numpy as np

ser1 = pd.Series([0,1,2])
print(ser1)
ser2 = pd.Series({'a':1,'b':2,'c':3})
print(ser2)
ser3 = pd.Series(np.arange(1,4))
print(ser3)