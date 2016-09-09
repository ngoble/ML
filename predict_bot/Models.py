from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from predict_bot.Massagers import Massagers
from predict_bot.Sigmoids import Normalizer

import numpy as np
import pandas as pd


class LSTM_Model(object):
    def __init__(self, look_back=1, neurons=4, dropout=0.2):
        self.model = Sequential()
        self.model.add(LSTM(neurons, input_dim=look_back, return_sequences=False, dropout_U=dropout, dropout_W=dropout))
        self.model.add(Dense(output_dim=1))
        self.model.add(Activation("linear"))
        self.model.compile(loss='mean_squared_error', optimizer='rmsprop')
        self.look_back = look_back

    def save(self, filepath):
        self.model.save(filepath)

    @staticmethod
    def load():
        return load_model('model.h5')

    def train_model(self, train_df, epochs=10, batch_size=1):

        train_data = Massagers.time_series_pd(train_df, look_back=self.look_back)

        unshaped_x = train_data[train_data.columns[:-1]].values
        unshaped_y = train_data[train_data.columns[-1]].values

        shaped_x = np.reshape(unshaped_x, (unshaped_x.shape[0], 1, unshaped_x.shape[1]))

        self.model.fit(shaped_x, unshaped_y, nb_epoch=epochs, batch_size=batch_size, verbose=1)

    def predict_with_known(self, predict_df):

        predict_data = Massagers.time_series_pd(predict_df, look_back=self.look_back)
        unshaped_x = predict_data[predict_data.columns[:-1]].values
        shaped_x = np.reshape(unshaped_x, (unshaped_x.shape[0], 1, unshaped_x.shape[1]))

        prediction = self.model.predict(shaped_x)
        actual = predict_data[predict_data.columns[-1]].values

        return prediction.flatten(), actual.flatten()

    def predict(self, predict_df):
        unshaped_x = predict_df.T.values
        shaped_x = np.reshape(unshaped_x, (unshaped_x.shape[0], 1, unshaped_x.shape[1]))
        prediction = self.model.predict(shaped_x)
        return prediction.flatten()

    def predict_np(self, predict_array):
        unshaped_x = predict_array
        shaped_x = np.reshape(unshaped_x, (unshaped_x.shape[0], 1, unshaped_x.shape[1]))
        prediction = self.model.predict(shaped_x)
        return prediction.flatten()


class Predictor(object):
    def __init__(self, historic_df, days_out=10, look_back=1, neurons=4, epochs=20):
        self.historic_df = historic_df
        self.days_out = days_out
        self.look_back = look_back
        self.normalizer = Normalizer(min_=0, max_=1)
        self.norm_df = self.normalizer.normalize(historic_df)

        self.lstm_model = LSTM_Model(look_back=self.look_back, neurons=neurons)
        self.lstm_model.train_model(self.norm_df, epochs=epochs)
        self.lstm_model.save('model.h5')
        # self.lstm_model = load_model('model.h5')
        self.residuals = self.get_residual_distribution()

    def get_residual_distribution(self):
        fit, actual = self.lstm_model.predict_with_known(self.norm_df)
        return fit - actual

    def randomize_window(self, window, num=1):
        mean = window.mean()
        random_index = np.random.choice(len(window), size=num, replace=False)
        random_values = np.random.choice(self.residuals, size=num, replace=True)
        for i, index in enumerate(random_index):
            window.iloc[index] = random_values[i] + mean
        return window

    def randomize_window_np(self, window):
        mean = np.mean(window, axis=1)
        random_index = np.random.choice(window.shape[1], size=window.shape[0], replace=True)
        random_values = np.random.choice(self.residuals, size=window.shape[0], replace=True)

        window[(np.arange(window.shape[0]), random_index)] = \
            np.reshape(random_values, (random_values.shape[0], 1)) + mean

        return window

    def predict_future_price(self):
        prediction = np.array([])
        window_df = self.norm_df.iloc[-self.look_back:]

        for i in range(self.days_out):

            result = self.lstm_model.predict(window_df)

            window_df = window_df.shift(-1)
            window_df.iloc[-1] = result
            window_df = self.randomize_window(window_df)

            prediction = np.append(prediction, result)

        prediction = self.normalizer.unnormalize(pd.DataFrame(prediction))

        return prediction

    def initialize_window(self, num_sims):
        window = np.array([self.norm_df.iloc[-self.look_back:].values])
        window = np.repeat(window, num_sims, axis=0)
        return window

    def predict_many_prices(self, num_sims):
        window = self.initialize_window(num_sims)
        prediction = np.zeros([num_sims, self.days_out])

        for i in range(self.days_out):

            result = self.lstm_model.predict_np(window)
            window = np.roll(window, -1)
            window[:, -1] = np.reshape(result, (result.shape[0], 1))
            window = self.randomize_window_np(window)

            prediction[:, i] = result

        prediction = self.normalizer.unnormalize(pd.DataFrame(prediction))

        return prediction




