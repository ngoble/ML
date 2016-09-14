import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import numpy as np

from predict_bot2.Models import Predictor
from predict_bot2.Plotters import Plotter

sns.set_style("dark")
np.random.seed(7)

days_out = 50
days_to_use = 365*3

raw_df = pd.read_csv('EURUSD.csv', header=0, index_col=0, parse_dates=True)
raw_hist_df = raw_df.ix[0:days_to_use]
raw_hist_df.reset_index(drop=True, inplace=True)
hist_data = raw_hist_df.values

predictor = Predictor(hist_data, days_out=days_out, neurons=200, look_back=days_out, validation_split=0.3,
                      epochs=200, dropout=0.3, plot_training=True)
# predictor.test_model(hist_data)

# print(50)

prediction_df = predictor.predict_many_prices(num_sims=2000)
pickle.dump(prediction_df, open("sims.p", "wb"))
# prediction_df = pickle.load(open("sims.p", "rb"))

plotter = Plotter()
plotter.plot_simulations(hist_data[-days_out*3:], prediction_df, num_paths_plotted=20)





