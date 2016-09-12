import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


class Plotter(object):

    @classmethod
    def plot_simulations(cls, historic, simulations):
        num_paths = 10
        highest_q = 0.99
        high_q = 0.95
        low_q = 0.05
        lowest_q = 0.01

        simulations = simulations.T
        simulations.index = np.arange(start=historic.index[-1] + 1,
                                      stop=historic.index[-1] + len(simulations) + 1)
        plt.plot(historic)
        plt.plot(simulations.iloc[:, :num_paths], color='#0e3e22', alpha=0.2, linewidth=1)

        upper_quantile = simulations.quantile(high_q, axis=1)
        lower_quantile = simulations.quantile(low_q, axis=1)
        highest_quantile = simulations.quantile(highest_q, axis=1)
        lowest_quantile = simulations.quantile(lowest_q, axis=1)

        plt.plot(upper_quantile, color='#0e3e22', alpha=0.7, linewidth=1.5)
        plt.plot(lower_quantile, color='#0e3e22', alpha=0.7, linewidth=1.5)
        plt.fill_between(simulations.index, upper_quantile, lower_quantile, alpha=0.4)

        # plt.plot(highest_quantile, color='#0e3e22', alpha=0.7, linewidth=1.5)
        # plt.plot(lowest_quantile, color='#0e3e22', alpha=0.7, linewidth=1.5)
        # plt.fill_between(simulations.index, highest_quantile, lowest_quantile, alpha=0.4)

        plt.plot(simulations.mean(axis=1), color='#e68a00', alpha=0.6, linewidth=2)



        plt.show()

        return 0