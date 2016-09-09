import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import seaborn as sns
import pickle

from predict_bot.Models import Predictor
from predict_bot.Plotters import Plotter

sns.set_style("dark")

raw_df = pd.read_csv('/Users/xuangao/PycharmProjects/ML/keras_play/data.csv', header=0, index_col=0, parse_dates=True)
# raw_df = raw_df.ix[0:400]
raw_df.reset_index(drop=True, inplace=True)

predictor = Predictor(raw_df, days_out=360)
#
prediction_df = predictor.predict_many_prices(num_sims=2000)
#
pickle.dump(prediction_df, open("sims.p", "wb"))
# prediction_df = pickle.load(open("sims.p", "rb"))

plotter = Plotter()
plotter.plot_simulations(raw_df.iloc[-720:], prediction_df)





