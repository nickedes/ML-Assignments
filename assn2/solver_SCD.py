import numpy as np
from scipy.sparse import csr_matrix
import sys
from sklearn.datasets import load_svmlight_file
import random
from datetime import datetime
import math
import matplotlib.pyplot as plt


def grad(w, Xtr, Ytr, i):
    gradient = (Ytr[i]*w*Xtr.getrow(i).T)[0, 0] - 1
    return gradient


def calculate_F(w, Xtr, Ytr):
    """
    """
    wx = (w*Xtr.T).T
    constraint = 0
    wx = wx.toarray()
    n = Xtr.get_shape()[0]
    for i in range(n):
        val = 1 - Ytr[i]*wx[i]
        if val > 0:
            constraint += val
    f = 0.5*(np.linalg.norm(w.toarray()))**2 + constraint
    return f


def draw_plots(time_elapsed, tick_vals, theotime_vals, obj_val):
    plt.plot(time_elapsed, obj_val, marker='o')
    plt.show()
    plt.plot(theotime_vals, obj_val, marker='x')
    plt.show()
    pass


def main():

    # Get training file name from the command line
    traindatafile = sys.argv[1]
    # For how many iterations do we wish to execute SCD?
    n_iter = int(sys.argv[2])
    # After how many iterations do we want to timestamp?
    spacing = int(sys.argv[3])

    # The training file is in libSVM format
    tr_data = load_svmlight_file(traindatafile)

    Xtr = tr_data[0]  # Training features in sparse format
    Ytr = tr_data[1]  # Training labels

    # We have n data points each in d-dimensions
    n, d = Xtr.get_shape()

    # The labels are named 1 and 2 in the data set. Convert them to our
    # standard -1 and 1 labels
    Ytr = 2*(Ytr - 1.5)
    Ytr = Ytr.astype(int)

    # Optional: densify the features matrix.
    # Warning: will slow down computations
    # Xtr = Xtr.toarray()

    # Initialize model
    # For dual SCD, you will need to maintain d_alpha and w
    # Note: if you have densified the Xt matrix then you can initialize w
    # as a NumPy array
    w = csr_matrix((1, d))
    d_alpha = np.zeros((n,))

    min_w = w

    # We will take a timestamp after every "spacing" iterations
    time_elapsed = np.zeros(math.ceil(n_iter/spacing))
    tick_vals = np.zeros(math.ceil(n_iter/spacing))
    theotime_vals = np.zeros(math.ceil(n_iter/spacing))
    obj_val = np.zeros(math.ceil(n_iter/spacing))

    tick = 0

    ttot = 0.0
    t_start = datetime.now()

    for t in range(n_iter):
        # Doing dual SCD
        # Choose a random coordinate from 1 to n
        i_rand = random.randint(1, n)
        
        # compute Gradient for random coordinate
        g = grad(w, Xtr, Ytr, i_rand)

        # projection step
        pg = g
        if d_alpha[i_rand] == 0:
            pg = min(g, 0)
        elif d_alpha[i_rand] == 1:
            pg = max(g, 0)

        if pg != 0:
            d_alpha_old = d_alpha[i_rand]
            Q = (Xtr.getrow(i_rand)*(Xtr.getrow(i_rand).T))[0, 0]
            d_alpha[i_rand] = min(max(d_alpha[i_rand] - g/Q, 0), 1)
            w = w + (d_alpha[i_rand] - d_alpha_old) * \
                Ytr[i_rand]*Xtr.getrow(i_rand)

        # Store the old and compute the new value of alpha along that
        # coordinate
        # d_alpha_old = d_alpha[i_rand]
        # d_alpha[i_rand] = min(
        #     max(d_alpha[i_rand] - grad(w, Xtr, Ytr, i_rand), 0), 1)
        # # Update the model - takes only O(d) time!
        # w = w + (d_alpha[i_rand] - d_alpha_old) * \
        #     Ytr[i_rand]*Xtr.getrow(i_rand)
        
        # Take a snapshot after every few iterations
        # Take snapshots after every spacing = 5000 or so SCD
        # iterations since they are fast
        if t % spacing == 0:
            # Stop the timer - we want to take a snapshot
            t_now = datetime.now()
            delta = t_now - t_start
            time_elapsed[tick] = ttot + delta.total_seconds()
            ttot = time_elapsed[tick]
            tick_vals[tick] = tick
            theotime_vals[tick] = tick_vals[tick]*spacing*d
            # Calculate the objective value f(w) for the current model w^t
            obj_val[tick] = calculate_F(w, Xtr, Ytr)
            if t == 0:
                min_f = obj_val[tick]

            if min_f > obj_val[tick]:
                min_f = obj_val[tick]
                min_w = w
            print(t, obj_val[tick])
            tick = tick+1
            # Start the timer again - training time!
            t_start = datetime.now()

    # draw_plots(time_elapsed, tick_vals, theotime_vals, obj_val)
    w_final = min_w.toarray()
    print("min f - ", min_f)
    np.save("model_SCD.npy", w_final)


if __name__ == '__main__':
    main()
