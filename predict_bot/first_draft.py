import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from predict_bot.Massagers import Massagers
from predict_bot.Sigmoids import Normalizer
from predict_bot.Models import LSTM_Model


n = np.array([[1,2,3],[2,3,4],[3,4,5]])
plt.plot(n)
plt.show()
exit()
np.random.seed(7)

look_back = 5
neurons = 8

# Get input data
raw_df = pd.read_csv('/Users/xuangao/PycharmProjects/ML/keras_play/data.csv', header=0, index_col=0, parse_dates=True)
raw_df = raw_df.ix[0:200]

# Normalize data between 0 and 1
normalizer = Normalizer(min_=0, max_=1)
norm_df = normalizer.normalize(raw_df)

# Split data into training and testing sets
train_df, test_df = Massagers.split_series_pd(norm_df, 0.60)

# Train the model
lstm_model = LSTM_Model(look_back=look_back, neurons=neurons)
lstm_model.train_model(train_df, epochs=100)

# Use the model to predict
print('gGOs')
outcomes = lstm_model.predict_with_known(train_df)
plt.plot(outcomes[0][:100])
plt.plot(outcomes[1][:100])
plt.show()


