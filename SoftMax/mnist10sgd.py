#----------------------------------------------------------------------------------------------------
#MNIST 10 Class Classification with Softmax Regression and Stocastic Gradient Descent
#By: Omar Hayat and Gabrielle Latreille
#----------------------------------------------------------------------------------------------------
import gzip
import pickle as pkl
import numpy as np
import random

var = [0,1,2,3,4,5,6,7,8,9]

def main():  
    train_set, test_set = getDataSets()
    n = 1 # batch size
    m = len(train_set[0])/n # number of batches
    alpha = 0.01

    params = [[0]*785]*len(var)

    temp = [[0]*785]*len(var)

    scores = [0]*1
    allsc = [-1]*2

    run_set = []

    for i in range(100):
        print("epoch: {}".format(i+1))
        temp = params
        run_set = randomize(train_set)
        for j in range(m):
            s = []
            hotset = []
            for k in range(j*n,(j+1)*n):
                s.append(softmax(run_set[0][k], temp))
                hotset.append(hotones(run_set[1][k]))

            temp -= alpha * costderiv(np.array(s), np.array(hotset), np.array(run_set[0][j*n:(j+1)*n]))
        sc = score(test_set, temp)
        # if sc <= scores[-1]:
        #     alpha /= 1.5
        #     print("bad: {}".format(sc))
        # else: 
        print("accuracy: {}".format(sc))
        params = temp
        scores.append(sc)
        if i%10 == 0:
            alpha*=0.5
        # if sc == allsc[-1] and sc == allsc[-2]:
        #     break
        allsc.append(sc)

    print(allsc)
    print(scores[-1])


def getDataSets():
    # unpack images and labels
    with gzip.open("mnist.pkl.gz", "rb") as f:
        set1, _set2, set3 = pkl.load(f)
        f.close()

    train_set = [[],[]]
    for x in range(len(set1[1])):
        if(set1[1][x] in var):
            train_set[0].append(np.insert(set1[0][x], 0, 1))
            train_set[1].append(set1[1][x])

    test_set = [[],[]]
    for x in range(len(set3[1])):
        if(set3[1][x] in var):
            test_set [0].append(np.insert(set3[0][x], 0, 1))
            test_set [1].append(set3[1][x])
    
    return [train_set,test_set]

def softmax(row, params):
    z = []
    for i in range(len(var)):
        z.append(np.e**(np.sum(row * params[i])))
    sm = []
    for i in range(len(var)):
        sm.append(z[i] / sum(z))
    return sm

def hotones(label):
    t = [0]*len(var)
    t[label] = 1
    return t

def costderiv(sm, ho, x):
    return (1/float(len(x)) * np.dot(x.T, (sm - ho))).T

def score(set, params):
    f = 0
    for i in range(len(set[0])):
        s = softmax(set[0][i], params)
        m = max(s)
        p = 1000
        for x in range(len(var)):
            if m == s[x]:
                p = x
        if p != set[1][i]:
            f+=1
    a = (len(set[0])-f)*100/float(len(set[0]))
    return a

def randomize(tset):
    tempset = copyarray(tset)
    nset = [[],[]]
    for _i in range(len(tempset[0])):
        r = random.randint(0,len(tempset[0])-1)
        nset[0].append(tempset[0][r])
        nset[1].append(tempset[1][r])
        tempset[0].pop(r)
        tempset[1].pop(r)
    return nset

def copyarray(arr):
    b = [[],[]]
    for i in range(len(arr[0])):
        b[0].append(arr[0][i])
        b[1].append(arr[1][i])
    return b

main()