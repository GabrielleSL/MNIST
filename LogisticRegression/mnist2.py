import gzip
import pickle as pkl
import numpy as np

var_1 = 6
var_2 = 8

def main():
    
    train_set, test_set = getDataSets()

    # thetas 0 to 784
    params = np.zeros(785) # size = 1 + num of feature = 785

    alpha = 0.1
    temp_params = []
    for j in range(len(params)):
        sum_num = 0
        for i in range(len(train_set[0])):
            if train_set[1][i] == var_1:
                k = 0
            else:
                k = 1
            sum_num += (hyp(train_set[0][i], params) - k) * train_set[0][i][j]
        temp_params.append(params[j] - alpha * sum_num) 
    
    for j in range(len(params)):
        params[j] = temp_params[j]

    # get the score
    score(test_set, params)

def getDataSets():
    # unpack images and labels
    with gzip.open("mnist.pkl.gz", "rb") as f:
        set1, _set2, set3 = pkl.load(f)
        f.close()

    # sort out the 0s and 1s of set1
    train_set = [[],[]]
    for x in range(len(set1[1])):
        if(set1[1][x] == var_1 or set1[1][x] == var_2):
            train_set[0].append(np.insert(set1[0][x], 0, 1))
            train_set[1].append(set1[1][x])

    # sort out the 0s and 1s of set3
    test_set = [[],[]]
    for x in range(len(set3[1])):
        if(set3[1][x] == var_1 or set3[1][x] == var_2):
            test_set [0].append(np.insert(set3[0][x], 0, 1))
            test_set [1].append(set3[1][x])
    
    return [train_set,test_set]

def hyp(row, params):
    z = np.sum(row * params)
    g = 1 / (1 + np.e**(-z))
    return g

def score(test_set, params):
    score = 0
    for i in range(len(test_set[1])):
        h = hyp(test_set[0][i], params)
        if(h>0.5):
            k = var_2
        else:
            k = var_1
        if k == test_set[1][i]:
            score += 1
    print(score, len(test_set[0]))

main()