import pandas as pd
import matplotlib.pyplot as plt
train_data = pd.read_csv('train.csv')
#review_scores_ratingが欠損値ではないデータを抜き出す
review_scores_rating_data = train_data[train_data['review_scores_rating'].notnull()]
review_scores_rating_data.plot.scatter(y='y',x='review_scores_rating')
plt.show()
review_scores_rating_data.plot.scatter(y='y',x='number_of_reviews')
plt.show()
