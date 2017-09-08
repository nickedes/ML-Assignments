# This code was used for K-NN Training file
from sklearn.datasets import load_svmlight_file
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import sys
import os.path


def kNearestNeighbor(X_train, y_train, X_test, k):
    # loop over all observations
    predictions = predict(X_train, y_train, X_test, k)
    return predictions


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


def run():

    Xtr, Ytr = gettrainData()

    Xts, Yts = gettestData()
    # Xtr, Ytr, Xts, Yts = Xtr[:10], Ytr[:10], Xts[:10], Yts[:10]
    # all values of k
    vals = [1, 2, 3, 5, 10]
    accuracy = {}
    test_error = {}
    # get predictions for all k
    predictions = kNearestNeighbor(Xtr, Ytr, Xts, vals)

    # evaluating accuracy
    for k in vals:
        correct_obs = np.sum(predictions[k] == Yts)
        num_test = Xts.shape[0]
        test_error[k] = num_test - correct_obs
        accuracy[k] = float(correct_obs) / num_test
        print('The accuracy of our classifier is ', accuracy)

    # save accuracies and error reported
    np.save('accuracyALLK1.npy', accuracy)
    np.save('errorALLK1.npy', test_error)
    print(accuracy, test_error)


def predict(Xtr, Ytr, Xts, vals, metric=None):
    '''
    Method: Calculate distance of test point from all training points and
    then get its label based on different k values
    '''
    N, D = Xtr.shape

    assert N == Ytr.shape[0], "Number of samples don't match"
    assert D == Xts.shape[1], "Train and test dimensions don't match"

    num_test = Xts.shape[0]
    num_train = Xtr.shape[0]
    dists = np.zeros((2, num_train))
    yts_labels = dict()
    for k in vals:
        yts_labels[k] = [0]*num_test
    for i in range(num_test):
        # calculate difference with each training point using broadcasting
        difference = Xtr[:, np.newaxis] - Xts[i]
        # get euclidean distance
        dists[0] = np.sqrt(np.square(difference).sum(axis=2)).T
        # take the training Ys
        dists[1] = Ytr
        transpose = dists.T
        predictions = transpose[transpose[:, 0].argsort()]
        y_predictions = predictions.T.astype(int)
        for k in vals:
            # Get the most common label among the k labels of neighbours
            yts_labels[k][i] = Counter(
                y_predictions[1][:k]).most_common(1)[0][0]
    return yts_labels


def plotAcc():
    accuracy = np.load('accuracyALLK1.npy').item()
    error = np.load('errorALLK1.npy').item()
    print(accuracy, error)
    x = np.array(list(accuracy.keys()))
    y = np.array(list(accuracy.values()))
    plt.plot(x, y, marker='o')
    plt.xlabel('k')
    plt.ylabel('Test Accuracy')

    plt.title("Test Accuracies vs k")

    plt.legend()
    plt.show()


def crossval():
    # 5-fold cross validation
    folds = 5

    print("cross", folds)
    # get training data
    Xtr, Ytr = gettrainData()
    # set of values of K for K-NN
    vals = [1, 2, 3, 5, 10, 15, 20, 100]

    # split list of number of training points into 5 parts (5-fold)
    split_data = np.array_split(range(Xtr.shape[0]), folds)

    # Create a list of splitted parts for Training data
    Xtr_fold = [Xtr[split_data[i]] for i in range(folds)]
    Ytr_fold = [Ytr[split_data[i]] for i in range(folds)]

    # save and continue, in case accuracy already stored!
    if os.path.isfile('5foldaccuracyAgain.npy'):
        accuracy = np.load('5foldaccuracyAgain.npy').item()
    else:
        accuracy = {}
    print(accuracy)

    # perform 5-fold cross validation

    for fold in range(folds):
        # All data is training data except the fold-th list
        train_Xts = np.concatenate(Xtr_fold[:fold] + Xtr_fold[fold + 1:])
        train_Yts = np.concatenate(Ytr_fold[:fold] + Ytr_fold[fold + 1:])

        # validation set
        validation_Xts = Xtr_fold[fold]
        validation_Yts = Ytr_fold[fold]

        # make Predictions!
        y_predictions = kNearestNeighbor(train_Xts, train_Yts,
                                         validation_Xts, vals)

        # evaluating accuracy
        for k in vals:
            correct_obs = np.sum(y_predictions[k] == validation_Yts)
            num_test = validation_Xts.shape[0]
            acc = float(correct_obs) / num_test
            accuracy[k] = accuracy.get(k, []) + [acc]
            np.save("5foldaccuracyAgain.npy", accuracy)

    # Print out the computed accuracies
    results = [(sum(accuracy[x])/len(accuracy[x]), x) for x in accuracy]

    # get value of k, with the largest accuracy
    tuned_K = sorted(results, key=lambda item: item[0], reverse=True)[0][1]
    return tuned_K


if __name__ == '__main__':
    # Run for finding accuracies for a set values of k
    # run()
    # plots Accuracies vs k
    # plotAcc()
    # Perform k-fold cross valdiation and reports best k
    crossval()
