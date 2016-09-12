import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import seaborn as sns
import pickle

from predict_bot2.Sigmoids import LogReturnsNormalizer
from predict_bot2.Models import Predictor
from predict_bot2.Plotters import Plotter

sns.set_style("dark")

days_out = 100

raw_df = pd.read_csv('data.csv', header=0, index_col=0, parse_dates=True)
raw_df = raw_df.ix[0:400]
raw_df.reset_index(drop=True, inplace=True)
hist_data = raw_df.values

predictor = Predictor(hist_data, days_out=days_out, neurons=200, look_back=days_out, validation_split=0.3, epochs=65,
                      dropout=0.8, plot_training=True)

print(predictor.test_model(hist_data))

prediction_df = predictor.predict_many_prices(num_sims=2000)
pickle.dump(prediction_df, open("sims.p", "wb"))
# prediction_df = pickle.load(open("sims.p", "rb"))

plotter = Plotter()
plotter.plot_simulations(hist_data[-days_out*2:], prediction_df, num_paths_plotted=20)





