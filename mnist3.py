import gzip
import pickle as pkl
import numpy as np
# import matplotlib.cm as cm
# import matplotlib.pyplot as plt

var = [0, 1, 2]

def main():
    
    train_set, test_set = getDataSets()

    # thetas 0 to 784
    params = [np.zeros(785), np.zeros(785), np.zeros(785)]

    alpha = 0.00001
    for x in range(3):
        temp_params = []
        # training
        for j in range(len(params[x])):
            sum_num = 0
            for i in range(3000):#len(train_set[0])):
                if train_set[1][i] == var[x]:
                    k = 1
                else:
                    k = 0
                sum_num += (hyp(train_set[0][i], params[x]) - k) * train_set[0][i][j]
            temp_params.append(params[x][j] - alpha * sum_num) 
        
        # update
        for j in range(len(params[x])):
            params[x][j] = temp_params[j]

    score(test_set, params)

def getDataSets():
    # unpack images and labels
    with gzip.open("mnist.pkl.gz", "rb") as f:
        set1, set2, set3 = pkl.load(f)
        f.close()

    # sort out the 0s and 1s of set1
    train_set = [[],[]]
    for x in range(len(set1[1])):
        if(set1[1][x] == var[0] or set1[1][x] == var[1] or set1[1][x] == var[2]):
            train_set[0].append(np.insert(set1[0][x], 0, 1))
            train_set[1].append(set1[1][x])

    # sort out the 0s and 1s of set3
    test_set = [[],[]]
    for x in range(len(set3[1])):
        if(set3[1][x] == var[0] or set3[1][x] == var[1] or set3[1][x] == var[2]):
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
        s_0 = hyp(test_set[0][i], params[0])
        s_1 = hyp(test_set[0][i], params[1])
        s_2 = hyp(test_set[0][i], params[2])
        m = max(s_0, s_1, s_2)
        if m == s_0:
            p = 0
        elif m == s_1:
            p = 1
        elif m == s_2:
            p = 2
        
        if p == test_set[1][i]:
            score += 1
    print(score)

main()