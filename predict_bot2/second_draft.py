import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import seaborn as sns
import pickle

from predict_bot2.Models import Predictor
from predict_bot2.Plotters import Plotter

sns.set_style("dark")

raw_df = pd.read_csv('/Users/xuangao/PycharmProjects/ML/keras_play/data.csv', header=0, index_col=0, parse_dates=True)
raw_df = raw_df.ix[0:400]
raw_df.reset_index(drop=True, inplace=True)
hist_data = raw_df.values

predictor = Predictor(raw_df, days_out=360, neurons=2, look_back=4, validation_split=0.3, epochs=100,
                      dropout=0.2, plot_training=True)

exit()

prediction_df = predictor.predict_many_prices(num_sims=2000)

pickle.dump(prediction_df, open("sims.p", "wb"))
# prediction_df = pickle.load(open("sims.p", "rb"))

plotter = Plotter()
plotter.plot_simulations(raw_df.iloc[-720:], prediction_df)





