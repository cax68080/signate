import pandas as pd
import matplotlib.pyplot as plt
train_data = pd.read_csv('train.csv')
train_data.plot.scatter(x='cleaning_fee',y='y')
plt.show()