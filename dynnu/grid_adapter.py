#!/usr/bin/env python3

import sys, getopt
print (sys.path[0])
sys.path.append('/home/mart/Documents/KybI/2019/python/hw-burgers')
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import numpy as np

__debug = False



def adapt_grid(weights, Xgc, deriv=0):
    # attempting to make GRID
    # dim of weights is (1 + deriv) less than that of Xgc
    targetLength = len(Xgc)
    if __debug:
        print('Target Length:', targetLength)
    weights = weights/np.sum(weights) * (len(Xgc) - 2) # two less than in grid
    if __debug:
        print("TotalWeights:", np.sum(weights))
    for i in range(deriv):
        Xgc = (Xgc[1:] + Xgc[:-1])/2 #mean
    i = 0
    # making sure not to have sections that need less than 1 grid point
    Xg2 = [0, ]
    w2 = []
    skip = 0
    leftOver = 0
    for w in weights:
        if skip > 0:
            skip -= 1
            i += 1
            continue
        if w < 1:
            upcoming = []
            for wc2 in weights[i + 1:]:
                if wc2 > 1:
                    break
                upcoming.append(wc2)
            total = w + np.sum(upcoming)
            final_index = i + len(upcoming)
            Xg2.append(Xgc[final_index + 1])
            leftOver += total - int(np.round(total))
            w2.append(int(np.round(total)))
            skip = len(upcoming)
            if __debug:
                print("%5.4f\t%5.4f\t%10s\t%5.4f\t(%5.4f)"%(Xgc[i], Xgc[final_index + 1], "->(NEW%d)"%(len(upcoming) + 1), w2[-1], leftOver))
        else:
            Xg2.append(Xgc[i + 1])
            leftOver += w - int(np.round(w))
            w2.append(int(np.round(w)))
            if __debug:
                print("%5.4f\t%5.4f\t%10s\t%5.4f\t(%5.4f)"%(Xgc[i], Xgc[i + 1], "->(REG)", w2[-1], leftOver))
        i += 1
    while leftOver > 0.9: # 
        # add to min
        w2[np.argmin(w2)] += 1
        leftOver -= 1
    if __debug:
        print("total w2:", np.sum(w2))
        print("weigths recalc:", w2)
        print("at ", len(Xg2), Xg2)
        print("start \tstop \ttotc \tw\tdone")
        print('LeftOVER:', leftOver)
    start = 0
    totc = 0
    Xgn = [Xg2[0],]
    done = 1
    # print(len(w2), len(Xg2))
    for w, stop in zip(w2, Xg2[1:]):
    # for w, stop in zip(weights, Xgc[1:]):
        totc += w
        if __debug:
            print("%3.4f\t%3.4f\t%3.4f\t%3.4f\t%d -> "%(start, stop, totc, w, done), end='')
        if totc >= 1:
            Xgn.append(np.linspace(start, stop, int(totc) + 1)[1:])
            if __debug:
                print(Xgn[-1])
            done += int(totc)
            totc -= int(totc)
            start = stop
        else:
            if __debug:
                print()
        if done >= targetLength - 2: # last to do
            Xgn.append((start + 1)/2)
            break
    if __debug:
        print('Leftover:', totc)
    Xgn = np.hstack(Xgn)
    if np.max(Xgn) < Xg2[-1]:
        Xgn = np.hstack((Xgn, 1))
    if __debug:
        print('new shape:', Xgn.shape)
    Xgn = np.sort(Xgn)        
    return Xgn

def adapt_grid_x(weights, Xgc):
    # adapt grid from derivative weights
    # dim of weights is 2 less than that of Xgc
    pass