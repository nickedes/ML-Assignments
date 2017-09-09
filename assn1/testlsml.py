from sklearn.datasets import load_svmlight_file
from collections import Counter
import numpy as np
import sys


def predict(Xtr, Ytr, Xts, metric=None):

    N, D = Xtr.shape

    assert N == Ytr.shape[0], "Number of samples don't match"
    assert D == Xts.shape[1], "Train and test dimensions don't match"

    if metric is None:
        metric = np.identity(D)

    Yts = np.zeros((Xts.shape[0], 1))
    dists = np.zeros((2, Xtr.shape[0]))
    # No. of nearest neighbours
    k = 10
    for i in range(Xts.shape[0]):
        '''
        Predict labels for test data using k-NN. Specify your tuned value of k here
        Calculate metric distance, the diagonal contains distances after performing matrix opns using broadcasting
        '''
        print("i = ", i)
        difference = Xtr - Xts[i]
        # product = difference * metric
        product = np.dot(difference, metric)
        # get only diagonal elements from result of product.difference.T
        dists[0] = np.einsum('ij,ji->i', product, difference.T)
        dists[1] = Ytr
        transpose = dists.T
        predictions = transpose[transpose[:, 0].argsort()]
        y_predictions = predictions.T.astype(int)
        Yts[i] = Counter(
            y_predictions[1][:k]).most_common(1)[0][0]
    return Yts


def main():

    # Get training and testing file names from the command line
    traindatafile = sys.argv[1]
    testdatafile = sys.argv[2]

    # The training file is in libSVM format
    tr_data = load_svmlight_file(traindatafile)

    Xtr = tr_data[0].toarray()
    Ytr = tr_data[1]

    # The testing file is in libSVM format too
    ts_data = load_svmlight_file(testdatafile)

    Xts = ts_data[0].toarray()
    # The test labels are useless for prediction. They are only used for
    # evaluation

    # Load the learned metric
    ms = ["modellsml4.npy"]

    for i in range(len(ms)):
        metric = np.load(ms[i])
        ### Do soemthing (if required) ###

        # Get the predicted values of labels
        Yts = predict(Xtr, Ytr, Xts, metric)
        # Save predictions to a file
        # Warning: do not change this file name
        if i == 0:
            np.savetxt("testlsml4.dat", Yts)


if __name__ == '__main__':
    main()
