import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

np.random.seed(7)


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i+look_back, 0])
    return np.array(dataX), np.array(dataY)

raw_df = pd.read_csv('/Users/xuangao/PycharmProjects/ML/keras_play/data.csv', header=0, index_col=0, parse_dates=True)
raw_df = raw_df.ix[0:20]

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(raw_df.values)

# split into train and test sets
train_size = int(len(dataset) * 0.5)
test_size = len(dataset) - train_size
train_data, test_data = dataset[0:train_size, :], dataset[train_size:, :]

look_back = 2
train_x, train_y = create_dataset(train_data, look_back)
test_x, test_y = create_dataset(test_data, look_back)

# reshape input to be [samples, time steps, features]
train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
print(train_x)
test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
print(train_y)
exit()

# make LSTM network
model = Sequential()
model.add(LSTM(4, input_dim=look_back))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='rmsprop')
model.fit(train_x, train_y, nb_epoch=10, batch_size=1, verbose=2)

# see how well the model performs
train_score = model.evaluate(train_x, train_y, verbose=0)
train_score = math.sqrt(train_score)
train_score = scaler.inverse_transform(np.array([[train_score]]))
print('Train Score: %.2f RMSE' % (train_score))
test_score = model.evaluate(test_x, test_y, verbose=0)
test_score = math.sqrt(test_score)
test_score = scaler.inverse_transform(np.array([[test_score]]))
print('Test Score: %.2f RMSE' % (test_score))

# generate predictions for training
trainPredict = model.predict(train_x)
testPredict = model.predict(test_x)

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

model.save('first_model.h5')

# plot baseline and predictions

# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
plt.show()


