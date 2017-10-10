import numpy as np
from scipy.sparse import csr_matrix
import sys
from sklearn.datasets import load_svmlight_file
from datetime import datetime
import math
import matplotlib.pyplot as plt


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

w = np.load('su/model_GD.npy')
wx = w*Xtr.T
print(np.sum(Ytr == np.sign(wx))/n)
# print(calculate_F(w, Xtr, Ytr))

w = np.load('su/model_SCD.npy')
print(w.shape)
wx = w*Xtr.T
print(np.sum(Ytr == np.sign(wx))/n)
# print(calculate_F(w, Xtr, Ytr))


# gdtime_elapsed = np.load("GDelap.npy")
# gdtheotime_vals = np.load("GDtheo.npy")
# gdobj_val = np.load("GDobj.npy")
# stime_elapsed = np.load("SCDelap.npy")
# stheotime_vals = np.load("SCDtheo.npy")
# sobj_val = np.load("SCDobj.npy")

# plt.plot(gdtime_elapsed, gdobj_val, marker='o')
# plt.plot(stime_elapsed, sobj_val, marker='x')

# plt.xlabel('Time elapsed --->')
# plt.ylabel('f(W) --->')
# plt.title('GD vs SCD')
# plt.legend(['GD', 'SCD'], loc='upper right', fontsize='small')
# plt.show()
# plt.close()


# plt.plot(stheotime_vals, sobj_val, marker='x')
# plt.plot(gdtheotime_vals, gdobj_val, marker='o')

# plt.xlabel('Theoretical time --->')
# plt.ylabel('f(W) --->')
# plt.title('GD vs SCD')
# plt.legend(['GD', 'SCD'], loc='upper right', fontsize='small')
# plt.show()
# plt.close()
# # print(gdtime_elapsed, stime_elapsed)


# plt.plot(gdtime_elapsed, gdobj_val, marker='o')
# plt.xlabel('Time elapsed --->')
# plt.ylabel('f(W) --->')
# plt.title('GD')
# plt.legend(['GD'], loc='upper right', fontsize='small')
# plt.show()
# plt.close()

# plt.plot(stime_elapsed, sobj_val, marker='x')
# plt.xlabel('Time elapsed --->')
# plt.ylabel('f(W) --->')
# plt.title('SCD')
# plt.legend(['SCD'], loc='upper right', fontsize='small')
# plt.show()
# plt.close()

# plt.plot(stheotime_vals, sobj_val, marker='x')
# plt.xlabel('Theoretical time --->')
# plt.ylabel('f(W) --->')
# plt.title('SCD')
# plt.legend(['SCD'], loc='upper right', fontsize='small')
# plt.show()
# plt.close()

# plt.plot(gdtheotime_vals, gdobj_val, marker='o')
# plt.xlabel('Theoretical time --->')
# plt.ylabel('f(W) --->')
# plt.title('GD')
# plt.legend(['GD'], loc='upper right', fontsize='small')
# plt.show()
# plt.close()
