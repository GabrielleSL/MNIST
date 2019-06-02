import gzip
import pickle as pkl
import numpy as np
# import matplotlib.cm as cm
# import matplotlib.pyplot as plt

var = [0, 1, 2, 3, 4]

def main():
    
    train_set, test_set = getDataSets()

    # thetas 0 to 784
    params = [np.zeros(785), np.zeros(785), np.zeros(785), np.zeros(785), np.zeros(785)]

    alpha = 0.00001
    for x in range(len(var)):
        temp_params = []
        # training
        for j in range(len(params[x])):
            # print("here")
            sum_num = 0
            for i in range(2000):#len(train_set[0])):
                if train_set[1][i] == var[x]:
                    k = 1
                else:
                    k = 0
                sum_num += (hyp(train_set[0][i], params[x]) - k) * train_set[0][i][j]
            temp_params.append(params[x][j] - alpha * sum_num) 
        
        # update
        for j in range(len(params[x])):
            params[x][j] = temp_params[j]

    # testing
    score(test_set, params)

def getDataSets():
    # unpack images and labels
    with gzip.open("mnist.pkl.gz", "rb") as f:
        set1, _set2, set3 = pkl.load(f)
        f.close()

    # sort out the 0s and 1s of set1
    train_set = [[],[]]
    for x in range(len(set1[1])):
        if(set1[1][x] in var):
            train_set[0].append(np.insert(set1[0][x], 0, 1))
            train_set[1].append(set1[1][x])

    # sort out the 0s and 1s of set3
    test_set = [[],[]]
    for x in range(len(set3[1])):
        if(set3[1][x] in var):
            test_set [0].append(np.insert(set3[0][x], 0, 1))
            test_set [1].append(set3[1][x])
    
    return [train_set,test_set]

def hyp(row, params):
    z = np.sum(row * params)
    g = 1 / (1 + np.e**(-z))
    return g

def score(test_set, params):
    score = 0

    for i in range(100):#len(test_set[1])):
        s = []
        for x in range(len(var)):
            s.append(hyp(test_set[0][i], params[x]))
        m = max(s)
        for x in range(len(var)):
            if m == s[x]:
                p = x
        
        if p == test_set[1][i]:
            score += 1
    print(score)

main()