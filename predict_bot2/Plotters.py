import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


class Plotter(object):

    @classmethod
    def plot_simulations(cls, historic, simulations, num_paths_plotted=10, high_q=95, low_q=5):

        simulations = simulations.T

        historic_index = np.arange(len(historic))
        simulated_index = np.arange(len(historic) - 1, len(historic) + len(simulations) - 1)

        plt.plot(historic_index, historic)
        plt.plot(simulated_index, simulations[:, :num_paths_plotted],
                 color='#0e3e22', alpha=0.2, linewidth=1)

        upper_quantile = np.percentile(simulations, high_q, axis=1)
        lower_quantile = np.percentile(simulations, low_q, axis=1)
        simulation_mean = np.mean(simulations, axis=1)

        plt.plot(simulated_index, upper_quantile, color='#0e3e22', alpha=0.7, linewidth=1.5)
        plt.plot(simulated_index, lower_quantile, color='#0e3e22', alpha=0.7, linewidth=1.5)
        plt.fill_between(simulated_index, upper_quantile, lower_quantile, alpha=0.4)

        plt.plot(simulated_index, simulation_mean, color='#e68a00', alpha=0.6, linewidth=2)

        plt.show()
