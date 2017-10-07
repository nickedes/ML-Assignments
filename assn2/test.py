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

# Get training file name from the command line
traindatafile = sys.argv[1]
# The training file is in libSVM format
tr_data = load_svmlight_file(traindatafile)

Xtr = tr_data[0]  # Training features in sparse format
Ytr = tr_data[1]  # Training labels
# We have n data points each in d-dimensions
n, d = Xtr.get_shape()
Xtr.toarray()
# The labels are named 1 and 2 in the data set. Convert them to our
# standard -1 and 1 labels
Ytr = 2*(Ytr - 1.5)
Ytr = Ytr.astype(int)
Ytr = Ytr.reshape(1, n)
w = np.load('model_SCD.npy')
wx = w*Xtr.T
print(wx)
print(np.sum(Ytr == np.sign(wx))/n)
print(calculate_F(w, Xtr, Ytr))
