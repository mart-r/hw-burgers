#!/usr/bin/env python3

import sys, getopt
print (sys.path[0])
sys.path.append('/home/mart/Documents/KybI/2019/python/hw-burgers')
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")
import numpy as np
from utils.nonuniform_grid import nonuniform_grid
from utils.init_utils import get_H_and_P
from hwbasics.HwBasics import Pnx_nu
from matplotlib import pyplot as plt


def do_test():
    J = 5
    X1, X1g = nonuniform_grid(J, 0.9)
    X2, X2g = nonuniform_grid(J, 1.1)
    X1, X2 = X1.reshape(1, 2*2**J), X2.reshape(1, 2*2**J)
    H_and_P1 = get_H_and_P(X1, X1g)
    H1, P11, P21, P2b1 = H_and_P1
    H_and_P2 = get_H_and_P(X2, X2g)
    H2, P12, P22, P2b2 = H_and_P2
    print(P21.shape, P2b1.shape, X1.shape)
    R2 = lambda X, P2, P2b: P2 - np.dot(P2b, X)
    R1 = lambda X, P1, P2b: P1 - np.dot(P2b, np.ones(X.shape))

    u1 = np.sin(2 * np.pi * X1)
    u2 = np.sin(2 * np.pi * X2)
    plt.plot(X1.flatten(), u1.flatten()),plt.plot(X2.flatten(), u2.flatten()), plt.legend(['u1', 'u2']),plt.show()

    # first try to get something from the middle
    A1 = np.linalg.lstsq(R2(X1, P21, P2b1).T, u1.T)[0].T
    print('max diff (back-forth) conversion:', np.max(np.abs(u1 - A1 @ R2(X1, P21, P2b1))))
    # mid
    xmid = np.random.rand() # RANDOM x
    Pmid2 = Pnx_nu(J, 2, xmid, X1g)
    umid = A1 @ R2(xmid, Pmid2, P2b1)
    umid = umid[0,0]
    print(xmid, umid)
    plt.plot(X1.flatten(), u1.flatten()), plt.plot(xmid, umid, 'o') ,plt.show()

    # the whole other one
    u1_2 = []
    for x in X2.flatten():
        Po2 = Pnx_nu(J, 2, x, X1g)
        u1_2.append((A1 @ R2(x, Po2, P2b1))[0, 0])
    u1_2 = np.array(u1_2)
    # Po2 = np.array(Po2)
    # print("Po2", Po2.shape)
    # u1_2 = A1 @ R2(X2, Po2, P2b1)
    plt.plot(X1.flatten(), u1.flatten()),plt.plot(X2.flatten(), u1_2.flatten()),plt.show()

    # all at once
    Po2 = []
    for x in X2.flatten():
        Po2.append(Pnx_nu(J, 2, x, X1g))#.flatten())

    Po2 = np.hstack(Po2)#.T
    print("Po2", Po2.shape)
    u1_2_2 = A1 @ R2(X2, Po2, P2b1)
    print(np.max(np.abs(u1_2 - u1_2_2)))
    plt.plot(X1.flatten(), u1.flatten()),plt.plot(X2.flatten(), u1_2_2.flatten()),plt.show()    
    


if __name__ == '__main__':
    do_test()