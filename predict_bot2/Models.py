import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from predict_bot2.Massagers import Massagers
from predict_bot2.Sigmoids import LogReturnsNormalizer

import numpy as np


class LSTM_Model(object):
    def __init__(self, look_back, neurons, dropout):
        self.model = Sequential()
        self.model.add(LSTM(neurons, input_dim=look_back, return_sequences=True))
        # self.model.add(Dropout(dropout))
        self.model.add(LSTM(neurons, return_sequences=False))
        self.model.add(Dense(output_dim=1))
        self.model.add(Activation("linear"))
        self.model.compile(loss='mean_squared_error', optimizer='rmsprop')
        self.look_back = look_back

    def save(self, filepath):
        self.model.save(filepath)

    @staticmethod
    def load():
        return load_model('model.h5')

    def train_model(self, train_array, validation_split, epochs, batch_size=1):

        x_train_data, y_train_data = Massagers.time_series(train_array, look_back=self.look_back)

        return self.model.fit(x_train_data, y_train_data, nb_epoch=epochs, batch_size=batch_size, verbose=2,
                              validation_split=validation_split)

    def evaluate_model(self, df):

        x_data, y_data = Massagers.time_series(df, look_back=self.look_back)
        score = self.model.predict(x_data)
        return y_data.flatten(), score

    def predict_np(self, predict_array):
        unshaped_x = predict_array
        shaped_x = np.reshape(unshaped_x, (unshaped_x.shape[0], 1, unshaped_x.shape[1]))
        prediction = self.model.predict(shaped_x)
        return prediction.flatten()


class Predictor(object):
    def __init__(self, historic_data, days_out=10, look_back=1, neurons=4, epochs=20, validation_split=0.3,
                 dropout=0.2, plot_training=False):
        self.historic_data = historic_data
        self.days_out = days_out
        self.look_back = look_back
        # self.normalizer = Normalizer(min_=0, max_=1)
        # self.normalized_data = self.normalizer.normalize(historic_data)
        self.normalizer = LogReturnsNormalizer(min_=0, max_=1)
        self.normalized_data = self.normalizer.normalize(self.historic_data)

        self.lstm_model = LSTM_Model(look_back=self.look_back, neurons=neurons, dropout=dropout)
        self.history = self.lstm_model.train_model(self.normalized_data, validation_split=validation_split, epochs=epochs)

        if plot_training:
            self.plot_learning()

        self.lstm_model.save('model.h5')
        # self.lstm_model = load_model('model.h5')
        self.residuals = self.normalized_data.flatten()

    def test_model(self, test_data):
        norm_test_data = self.normalizer.normalize(test_data)
        actual, fitted = self.lstm_model.evaluate_model(norm_test_data)
        fitted = fitted.flatten()

        actual = self.normalizer.unnormalize_array(actual)
        fitted = self.normalizer.unnormalize_array(fitted)

        plt.plot(actual)
        plt.plot(fitted)
        plt.show()

    def plot_learning(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def randomize_window(self, window, num=1):
        mean = window.mean()
        random_index = np.random.choice(len(window), size=num, replace=False)
        random_values = np.random.choice(self.residuals, size=num, replace=True)
        for i, index in enumerate(random_index):
            window.iloc[index] = random_values[i] + mean
        return window

    def randomize_window_np(self, window):
        random_index = np.random.choice(window.shape[1], size=window.shape[0], replace=True)
        random_values = np.random.choice(self.residuals, size=window.shape[0], replace=True)

        window[(np.arange(window.shape[0]), random_index)] = \
            np.reshape(random_values, (random_values.shape[0], 1))

        return window

    def initialize_window(self, num_sims):
        window = np.array([self.normalized_data[-self.look_back:]])
        window = np.repeat(window, num_sims, axis=0)
        return window

    def predict_many_prices(self, num_sims):
        window = self.initialize_window(num_sims)
        prediction = np.zeros([num_sims, self.days_out+1])
        prediction[:, 0] = window[:, -1].flatten()

        for i in range(1, self.days_out + 1):

            result = self.lstm_model.predict_np(window)
            window = np.roll(window, -1)
            window[:, -1] = np.reshape(result, (result.shape[0], 1))
            window = self.randomize_window_np(window)

            prediction[:, i] = result

        prediction = self.normalizer.unnormalize(prediction)

        return prediction




