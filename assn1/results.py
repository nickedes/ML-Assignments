import numpy as np
from train import gettestData

pred_Y = np.loadtxt("testY.dat")
Xts, Yts = gettestData()

correct_obs = np.sum(pred_Y == Yts)
num_test = Xts.shape[0]
acc = float(correct_obs) / num_test
print(acc)

# accuracy = np.load('5foldaccuracyAgain.npy').item()

# # Print out the computed accuracies
# results = [(sum(accuracy[x])/len(accuracy[x]), x) for x in accuracy]

# # get value of k, with the largest accuracy
# tuned_K = sorted(results, key=lambda item: item[0], reverse=True)[0][1]

# for x in accuracy:
#     print(x, accuracy[x])

# for x in accuracy:
#     print(x, "average - ", sum(accuracy[x])/len(accuracy[x]))

# print(tuned_K)
