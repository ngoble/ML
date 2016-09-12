import numpy as np
import pandas as pd


class Massagers(object):

    @staticmethod
    def time_series(data, look_back=1):

        final_array = np.zeros((len(data), look_back+1))
        final_array[0:, 0] = data[:, 0]

        for i in range(1, look_back+1):
            final_array[i:, i] = data[:-i, 0]

        final_array = final_array[look_back:, :]

        x_array = final_array[:, :look_back]
        x_array = np.reshape(x_array, (x_array.shape[0], 1, x_array.shape[1]))
        y_array = final_array[:, look_back]

        return x_array, y_array

    @staticmethod
    def split_series_pd(data, split_level):
        return data.iloc[:int(len(data)*split_level)], data.iloc[int(len(data)*split_level):]
