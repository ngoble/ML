from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


class Normalizer(object):
    def __init__(self, min_, max_):
        self.min_ = min_
        self.max_ = max_
        self.scaler = MinMaxScaler(feature_range=(self.min_, self.max_))

    def normalize(self, data):
        return self.scaler.fit_transform(data)

    def unnormalize(self, data):
        return self.scaler.inverse_transform(data)


