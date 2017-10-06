import numpy as np
from scipy.sparse import csr_matrix
import sys
from sklearn.datasets import load_svmlight_file
from datetime import datetime
import math


def calculate_F(w, Xtr, Ytr):
    """
    """
    w = csr_matrix(w)
    wx = csr_matrix.dot(w, Xtr.T)
    ywx = wx.multiply(Ytr)
    constraint = 0
    z = (ywx < 1).toarray()
    constraint = (1 - ywx.toarray()[z]).sum(axis=0)

    f = 0.5*(np.linalg.norm(w.toarray()))**2 + constraint
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
    # Optional: densify the features matrix.
    # Warning: will slow down computations
    # Xtr = Xtr.toarray()

    # Initialize model
    # For primal GD, you only need to maintain w
    # Note: if you have densified the Xt matrix then you can initialize w as a
    # NumPy array
    w = csr_matrix((1, d))
    wbar = csr_matrix((1, d))

    # We will take a timestamp after every "spacing" iterations
    time_elapsed = np.zeros(math.ceil(n_iter/spacing))
    tick_vals = np.zeros(math.ceil(n_iter/spacing))
    obj_val = np.zeros(math.ceil(n_iter/spacing))

    tick = 0

    ttot = 0.0
    t_start = datetime.now()
    for t in range(1, n_iter+1):
        try:
            # Doing primal GD

            # Compute gradient
            w = csr_matrix(w)
            wx = csr_matrix.dot(w, Xtr.T)
            ywx = wx.multiply(Ytr)
            yx = Xtr.multiply(Ytr.T)
            z = (ywx < 1).toarray().ravel()
            val = np.sum(yx.toarray()[z], axis=0)
            g = w - val
            g.reshape(1, d)  # Reshaping since model is a row vector

            # Calculate step lenght. Step length may depend on n and t
            C = 0.5 * 10**(2)
            eta = C/math.sqrt(t)

            # Update the model
            w = w - eta * g/n

            # Use the averaged model if that works better (see [\textbf{SSBD}]
            # section 14.3)
            wbar = (wbar*(t-1) + w)/t

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
                # Calculate the objective value f(w) for the current model w^t or
                # the current averaged model \bar{w}^t
                obj_val[tick] = calculate_F(w, Xtr, Ytr)
                print(t, obj_val[tick], len(str(int(obj_val[tick]))))
                tick = tick+1
                # Start the timer again - training time!
                t_start = datetime.now()
        except KeyboardInterrupt:
            break

    # Choose one of the two based on whichever works better for you
    w_final = np.array(w)
    # w_final = wbar.toarray()
    np.save("model_GD.npy", w_final)


if __name__ == '__main__':
    main()
