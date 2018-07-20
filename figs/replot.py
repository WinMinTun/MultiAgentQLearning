"""
replot data
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
font = {'size': 15}
rc('font', **font)


def plot_uncut(data_file):
    output = pd.read_csv(os.path.join("{}.csv".format(data_file)), header=0, index_col=0)
    ax = output[["Q-value Difference"]].plot(legend=False)
    ax.set_xlabel("Simulation Iteration")
    ax.set_ylabel("Q-value Difference")
    #ax.set_ylim(bottom=0, top=0.5)
    #plt.show()
    plt.savefig(os.path.join("{}_uncut.png".format(data_file)))

def plot_cut(data_file):
    output = pd.read_csv(os.path.join("{}.csv".format(data_file)), header=0, index_col=0)
    ax = output[["Q-value Difference"]].plot(legend=False)
    ax.set_xlabel("Simulation Iteration")
    ax.set_ylabel("Q-value Difference")
    #ax.set_ylim(bottom=0, top=0.5)
    #plt.show()
    plt.savefig(os.path.join("{}_uncut.png".format(data_file)))




if __name__ == '__main__':

    print("this is the code for the saving results and linear programming for maxmin and ce")

    data_file = sys.argv[1]
    plot_uncut(data_file)




