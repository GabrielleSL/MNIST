import gzip
import pickle as pkl
import numpy as np

var = np.arange(10)

def main():
    
    train_set, test_set = getDataSets()

    # thetas 0 to 784
    params = [[0]*785]*10
    
    print(len(var))
    alpha = 0.00001
    for x in range(len(var)):
        temp_params = []
        # training with logistic regression gradient descent
        for j in range(len(params[x])): #loops over all paramaters 
            sum_num = 0
            for i in range(1000):#len(train_set[0])): loops over training set, sum part of function
                if train_set[1][i] == var[x]:
                    k = 1
                else:
                    k = 0
                sum_num += (hyp(train_set[0][i], params[x]) - k) * train_set[0][i][j]
                #Simultaneously update all of the paramaters
            temp_params.append(params[x][j] - alpha * sum_num) 
        
        #take one step in gradient descent to update paramaters
        
        # update
        for j in range(len(params[x])):
            params[x][j] = temp_params[j]

    # testing
    score(test_set, params)

def getDataSets():
    # unpack images and labels
    with gzip.open("mnist.pkl.gz", "rb") as f:
        set1, set2, set3 = pkl.load(f)
        f.close()

    #append column of 1's to training set
    train_set = [[],[]]
    for x in range(len(set1[1])):
        if(set1[1][x] in var):
            train_set[0].append(np.insert(set1[0][x], 0, 1))
            train_set[1].append(set1[1][x])

    #append column of 1's to test set
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
    
    for i in range(len(test_set[1])):
        s = []
        for x in range(len(var)):
            s.append(hyp(test_set[0][i], params[x]))
        m = max(s)
        for x in range(len(var)):
            if m == s[x]:
                p = x
        
        if p == test_set[1][i]:
            score += 1
    print(score, len(test_set[1]))

main()