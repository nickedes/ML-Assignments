import numpy as np
from train import gettestData

pred_Y = np.loadtxt("testY.dat")
Xts, Yts = gettestData()

correct_obs = np.sum(pred_Y == Yts)
num_test = Xts.shape[0]
acc = float(correct_obs) / num_test
print(acc)
