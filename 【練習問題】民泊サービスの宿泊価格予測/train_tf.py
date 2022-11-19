import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
train_data = pd.read_csv('train.csv')

sns.boxplot(data=train_data,x='cleaning_fee',y='y')
plt.ylim(0,600)
plt.show()
sns.boxplot(data=train_data,x='host_identity_verified',y='y')
plt.ylim(0,600)
plt.show()
sns.boxplot(data=train_data,x='instant_bookable',y='y')
plt.ylim(0,600)
plt.show()
