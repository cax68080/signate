#データの読み込みと整形
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df_test = pd.read_csv(".\\SIGNATE\\健診データによる肝疾患判定\\test.csv")
print(df_test.head())
print(df_test.shape)
