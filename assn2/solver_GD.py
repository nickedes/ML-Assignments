import numpy as np
from scipy.sparse import csr_matrix
import sys
from sklearn.datasets import load_svmlight_file
from datetime import datetime
import math


def calculate_F(w, Xtr, Ytr):
    """
        calculate value of primal objective
    """
    w = csr_matrix(w)
    wx = csr_matrix.dot(w, Xtr.T)
    ywx = wx.multiply(Ytr)
    # calculate sum of slack variables
    slackSum = (1 - ywx.toarray()[(ywx < 1).toarray()]).sum(axis=0)

    f = 0.5*(np.linalg.norm(w.toarray()))**2 + slackSum
    return f


def main():

    # Get training file name from the command line
    traindatafile = sys.argv[1]

    # For how many iterations do we wish to execute GD?
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
    Ytr = Ytr.reshape(1, n)

    # Initialize model
    # For primal GD, you only need to maintain w
    w = csr_matrix((1, d))

    # We will take a timestamp after every "spacing" iterations
    time_elapsed = np.zeros(math.ceil(n_iter/spacing))
    tick_vals = np.zeros(math.ceil(n_iter/spacing))
    theotime_vals = np.zeros(math.ceil(n_iter/spacing))
    obj_val = np.zeros(math.ceil(n_iter/spacing))

    tick = 0

    ttot = 0.0
    t_start = datetime.now()
    # constant used in step length
    C = 0.3 * 10**(2)
    for t in range(0, n_iter):
        try:
            # Doing primal GD

            # Compute gradient
            w = csr_matrix(w)
            # calculate <w, X^i> for all i
            wx = csr_matrix.dot(w, Xtr.T)
            # calculate y^i * <w, X^i> for all i
            ywx = wx.multiply(Ytr)
            # calculate y^i * X^i for all i
            yx = Xtr.multiply(Ytr.T)
            # Those y^i * <w, X^i> which are less than 1
            condition = (ywx < 1).toarray().ravel()
            # sum of all y^i * X^i where y^i * <w, X^i> is less than 1
            val = np.sum(yx.toarray()[condition], axis=0)
            # evaluate gradient
            g = w - val
            g.reshape(1, d)  # Reshaping since model is a row vector

            # step lenght. Step length depends on n and t
            eta = C/(n*math.sqrt(t+1))

            # Update the model
            w = w - eta * g

            # Take a snapshot after every few iterations
            # Take snapshots after every spacing = 5 or 10 GD iterations since they
            # are slow
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
                tick = tick+1
                # Start the timer again - training time!
                t_start = datetime.now()
        except KeyboardInterrupt:
            break

    # Choosen w over wbar
    w_final = np.array(w)
    # save model
    np.save("model_GD.npy", w_final)
    np.save("GDelap.npy", time_elapsed)
    np.save("GDtheo.npy", theotime_vals)
    np.save("GDobj.npy", obj_val)


if __name__ == '__main__':
    main()
