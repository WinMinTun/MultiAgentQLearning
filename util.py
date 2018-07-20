"""
helper function for
saving results to csv and plotting, and
linear programming to calculate minimax and CE
"""
import os
import sys
import pandas as pd
import numpy as np
from cvxopt import matrix, solvers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
font = {'size': 15}
rc('font', **font)
solvers.options['show_progress'] = False
solvers.options['glpk'] = {'tm_lim': 1000} # max timeout for glpk
solvers.options['show_progress'] = False # disable solver output
solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}  # cvxopt 1.1.8
solvers.options['LPX_K_MSGLEV'] = 0  # previous versions

def categorical_sample(prob_n):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    prob_n = prob_n / prob_n.sum()
    #csprob_n = np.cumsum(prob_n)
    return np.random.choice(range(prob_n.shape[0]), p=prob_n)

def re_plot(data_file):
    output = pd.read_csv(os.path.join("results", "{}.csv".format(data_file)), header=0, index_col=0)
    ax1 = output[["Q-value Difference"]].plot(legend=False)
    ax1.set_xlabel("Simulation Iteration")
    ax1.set_ylabel("Q-value Difference")
    ax1.set_ylim(bottom=0, top=0.5)
    #plt.show()
    plt.savefig(os.path.join("results", "{}_cut.png".format(data_file)))
    plt.close()

    ax2 = output[["Q-value Difference"]].plot(legend=False)
    ax2.set_xlabel("Simulation Iteration")
    ax2.set_ylabel("Q-value Difference")
    #ax2.set_ylim(bottom=0, top=0.5)
    #plt.show()
    plt.savefig(os.path.join("results", "{}_uncut.png".format(data_file)))
    plt.close()

def log(learner_name, alpha, alpha_end, epsilon, epsilon_end, maxepisode, solver):
    logfile_name = os.path.join("results", "log.csv")
    if os.path.isfile(logfile_name):
        log = pd.read_csv(logfile_name, index_col=0, header=0)
        trial_num = max(list(log.index)) + 1
    else:
        trial_num = 0
        log = pd.DataFrame(columns=["learner_name", "alpha", "alpha_end", "epsilon", "epsilon_end", "maxepisode", "solver"])
    #entry = {"learner_name":[learner_name], "alpha":[alpha], "alpha_end":[alpha_end], "epsilon":[epsilon], "epsilon_end":[epsilon_end], "maxepisode":[maxepisode], "solver":[solver]}
    entry = [learner_name, alpha, alpha_end, epsilon, epsilon_end, maxepisode, solver]
    #entry = pd.DataFrame(entry)
    #print(list(entry.index))
    log.loc[trial_num] = entry
    #print(log)
    log.to_csv(logfile_name)
    return trial_num


def save_results(data, final_policy, Qtable, trial_num):
    output = pd.DataFrame(data=np.array(data), columns=["Simulation Iteration", "Q-Value"])
    output.set_index("Simulation Iteration", inplace=True)
    output["Q-value Difference"] = np.abs(output.diff(periods=1, axis=0))

    output.to_csv(os.path.join("results", "Qdifference_{}.csv".format(trial_num)))
    ax = output[["Q-value Difference"]].plot(legend=False)
    ax.set_xlabel("Simulation Iteration")
    ax.set_ylabel("Q-value Difference")
    ax.set_ylim(bottom=0, top=0.5)
    #plt.show()
    plt.savefig(os.path.join("results", "Qdifference_{}_cut.png".format(trial_num)))
    plt.close()

    ax2 = output[["Q-value Difference"]].plot(legend=False)
    ax2.set_xlabel("Simulation Iteration")
    ax2.set_ylabel("Q-value Difference")
    #ax2.set_ylim(bottom=0, top=0.5)
    #plt.show()
    plt.savefig(os.path.join("results", "Qdifference_{}_uncut.png".format(trial_num)))
    plt.close()

    df_final_policy = pd.DataFrame(final_policy)
    df_final_policy.to_csv(os.path.join("results", "final_policy_{}.txt".format(trial_num)), header=False, index=False)

    np.save(os.path.join("results", "Qtable_{}.npy".format(trial_num)), Qtable)


def maxmin(A, solver="glpk"):
    num_vars = len(A)
    # minimize matrix c: minimize c*x
    c = [-1] + [0] * num_vars
    c = np.array(c, dtype="float")
    c = matrix(c)
    # constraints G*x <= h
    G = np.matrix(A, dtype="float").T # reformat each variable is in a row
    G *= -1 # minimization constraint
    G = np.vstack([G, np.eye(num_vars) * -1]) # > 0 constraint for all vars
    new_col = [1] * num_vars + [0] * num_vars
    G = np.insert(G, 0, new_col, axis=1) # insert utility column
    G = matrix(G)
    h = [0] * num_vars * 2
    h = np.array(h, dtype="float")
    h = matrix(h)
    # contraints Ax = b -> sum of all probabilites is 1
    A = [0] + [1] * num_vars
    A = np.matrix(A, dtype="float")
    A = matrix(A)
    b = np.matrix(1, dtype="float")
    b = matrix(b)
    sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver=solver)
    return sol

def ce(A, solver="glpk"): #correlated equilibrium
    num_vars = len(A)
    # maximize matrix c
    c = [sum(i) for i in A] # sum of payoffs for both players
    c = np.array(c, dtype="float")
    c = matrix(c)
    c *= -1 # cvxopt minimizes so *-1 to maximize the sum of both players' reward
    # constraints G*x <= h
    G = build_ce_constraints(A=A)
    G = np.vstack([G, np.eye(num_vars) * -1]) # > 0 constraint for all vars
    h_size = len(G)
    G = matrix(G)
    h = [0 for i in range(h_size)]
    h = np.array(h, dtype="float")
    h = matrix(h)
    # contraints Ax = b
    A = [1 for i in range(num_vars)]
    A = np.matrix(A, dtype="float")
    A = matrix(A)
    b = np.matrix(1, dtype="float")
    b = matrix(b)
    sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver=solver)
    return sol

def build_ce_constraints(A):
    num_vars = int(len(A) ** (1/2))
    G = []
    # row player
    for i in range(num_vars): # action row i
        for j in range(num_vars): # action row j
            if i != j:
                constraints = [0 for i in A]
                base_idx = i * num_vars
                comp_idx = j * num_vars
                for k in range(num_vars):
                    constraints[base_idx+k] = (- A[base_idx+k][0]
                                               + A[comp_idx+k][0])
                G += [constraints]
    # col player
    for i in range(num_vars): # action column i
        for j in range(num_vars): # action column j
            if i != j:
                constraints = [0 for i in A]
                for k in range(num_vars):
                    constraints[i + (k * num_vars)] = (
                        - A[i + (k * num_vars)][1] 
                        + A[j + (k * num_vars)][1])
                G += [constraints]
    return np.matrix(G, dtype="float")


if __name__ == '__main__':

    print("this is the code for the saving results and linear programming for maxmin and ce")

    data_file = sys.argv[1]
    re_plot(data_file)




