from sklearn.preprocessing import MinMaxScaler
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


class LogReturnsNormalizer(object):
    def __init__(self, min_, max_):
        self.min_ = min_
        self.max_ = max_
        self.scaler = MinMaxScaler(feature_range=(self.min_, self.max_))
        self.last_value = np.empty([])

    def normalize(self, data):
        log_returns = self.rates_to_log_returns(data)
        self.last_value = data[-1]
        return self.scaler.fit_transform(log_returns)

    def unnormalize(self, norm_log_returns):
        log_returns = self.scaler.inverse_transform(norm_log_returns)

        simulated_rates = np.empty((np.shape(log_returns)))
        simulated_rates[:, 0] = np.repeat(self.last_value, log_returns.shape[0])

        for row in range(1, log_returns.shape[1]):
            simulated_rates[:, row] = self.log_return_to_rate(log_returns[:, row], simulated_rates[:, row-1])

        return simulated_rates

    @staticmethod
    def rates_to_log_returns(rates):
        log_returns = np.log(rates[1:] / rates[:-1])
        if np.isnan(log_returns).any():
            print("SOMETHING WENT WRONG WITH LOG_RETURNS CONVERSION")
        return log_returns

    @staticmethod
    def log_return_to_rate(log_return, prior_rate):
        return np.exp(log_return) * prior_rate



