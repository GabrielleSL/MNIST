import numpy as np
import gzip
import pickle as pkl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as lg
#import theano

# open and unpickle file
with gzip.open("mnist.pkl.gz", "rb") as f:
    train_set, valid_set, test_set = pkl.load(f)
    f.close()

print(train_set[0][0])

# sort out the 0s and 1s of train set
bin_set = train_set
# for x in range(50000):
#     if(train_set[1][x] <=1):
#         bin_set[0].append(train_set[0][x])
#         bin_set[1].append(train_set[1][x])

# sort out the 0s and 1s of test set
bin_test_set = test_set
# for x in range(10000):
#     if(test_set[1][x] <=1):
#         bin_test_set [0].append(test_set[0][x])
#         bin_test_set [1].append(test_set[1][x])

# show the image and label
#   for x in range(0,10):
#     print(bin_set[1][x])
    #plt.imshow(bin_set[0][x].reshape((28, 28)), cmap=cm.Greys_r)
    #plt.show()  
    
logReg = lg(solver = "lbfgs", multi_class="auto", max_iter=50000)

# print(len(bin_test_set[0]))

# print("part 1")

logReg.fit(bin_set[0], bin_set[1])

# print("part 2")

# # Predict one image
# for x in range(2115):
#     pred = logReg.predict(bin_test_set[0][x].reshape(1,-1))
#     if(pred != bin_test_set[1][x]):
#         print("prediction: {}".format(pred))
#         print("actual: {}".format(bin_test_set[1][x]))
#         plt.imshow(bin_test_set[0][x].reshape((28, 28)), cmap=cm.Greys_r)
#         plt.show()

# print(bin_set[0][0])
score = logReg.score(bin_test_set[0],bin_test_set[1])

print("score: {}".format(score))
