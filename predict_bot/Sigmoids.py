from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class Normalizer(object):
    def __init__(self, min_, max_):
        self.min_ = min_
        self.max_ = max_
        self.scaler = MinMaxScaler(feature_range=(self.min_, self.max_))

    def normalize(self, data):
        normalized_data = self.scaler.fit_transform(data.values)
        return pd.DataFrame(normalized_data, index=data.index, columns=data.columns)

    def unnormalize(self, data):
        unnormalized_data = self.scaler.inverse_transform(data.values)
        return pd.DataFrame(unnormalized_data, index=data.index, columns=data.columns)

