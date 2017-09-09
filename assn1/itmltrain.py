from sklearn.datasets import load_svmlight_file
from metric_learn import ITML_Supervised
import numpy as np
import sys


def gettrainData():
    # Get training file name from the command line
    traindatafile = sys.argv[1]

    # The training file is in libSVM format
    tr_data = load_svmlight_file(traindatafile)

    Xtr = tr_data[0].toarray()  # Converts sparse matrices to dense
    Ytr = tr_data[1]  # The trainig labels
    return Xtr, Ytr


def gettestData():

    # Get testing file name from the command line
    testdatafile = sys.argv[2]

    # The testing file is in libSVM format
    ts_data = load_svmlight_file(testdatafile)

    Xts = ts_data[0].toarray()  # Converts sparse matrices to dense
    Yts = ts_data[1]  # The trainig labels
    return Xts, Yts


# get training data
Xtr, Ytr = gettrainData()
# get testing data
Xts, Yts = gettestData()

# Taking only a fraction of data. i.e. 1/4th
Xtr = Xtr[:len(Xtr)//4]
Ytr = Ytr[:len(Ytr)//4]

itml = ITML_Supervised(num_constraints=1000)
# learning
itml.fit(Xtr, Ytr)
# Get the learnt metric
M = itml.metric()

# Metric saved
np.save("model.npy", M)
