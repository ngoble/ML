import numpy as np
import pandas as pd


class Massagers(object):

    @staticmethod
    def time_series_pd(data, look_back=1):
        df = pd.DataFrame(columns=np.append(np.arange(0, look_back), 'y'))
        df['0'] = data[data.columns[0]].values
        for i in range(1, look_back+1):
            df[df.columns[i]] = df[df.columns[i-1]].shift(-1)
        return df.dropna()

    @staticmethod
    def time_series_np(data, look_back=1):
        data = pd.DataFrame(data)
        df = pd.DataFrame(columns=np.append(np.arange(0, look_back), 'y'))
        df['0'] = data[data.columns[0]].values
        for i in range(1, look_back+1):
            df[df.columns[i]] = df[df.columns[i-1]].shift(-1)
        return df.dropna()

    @staticmethod
    def split_series_pd(data, split_level):
        return data.iloc[:int(len(data)*split_level)], data.iloc[int(len(data)*split_level):]
