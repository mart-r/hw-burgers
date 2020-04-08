#!/usr/bin/env python3

import numpy as np
import sys
# integration
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz as integrate

# my "package"
sys.path.append('/home/mart/Documents/KybI/2019/python/hw-burgers')

# from utils.nonuniform_grid import nonuniform_grid
from utils.utils import plot_grid


def derive_grid(Xg, w12, bPrint=False):#, span=1):
    # dx = np.diff(Xg)
    wi = 1/w12
    w = wi/np.sum(wi)
    N = len(Xg)
    A = np.zeros((N - 1, N - 2))
    b = np.zeros((N - 1, 1))
    for k in range(N-1):
        if k == 0:
            b[k] = 0 # x0
        else:
            A[k, k - 1] = -1 # k
        if k == N - 2:
            b[k] = -1 # -xN
        else:
            A[k, k] = 1      # k + 1
        b[k] += w[k]
    if bPrint:
        print("%5d"*len(A[0])%tuple(range(len(A[0]))))
        for line, rhs in zip(A, b.flatten()):
            cline = line[:]
            # cline[cline != 0] = cline[cline != 0]/np.abs(cline[cline != 0])
            print(("%5.1f"*len(line))%tuple(cline), "|%g"%rhs)
    Xmid = np.linalg.lstsq(A,b)[0].flatten()
    Xg2 = np.hstack((0, Xmid, 1))
    if bPrint:
        print('Xg2', Xg2)
    return Xg2, (Xg2[1:] + Xg2[:-1])/2.

def iterate_derivation(fun, Xg0, interp, maxit=10, diffTol=0.01, bVerbose=False):
    Xg2 = np.array(Xg0) # make copy
    X2  = (Xg2[1:] + Xg2[:-1])/2.
    w2 = np.array(interp(X2))
    map = {}
    smallestI = 0
    smallestDiff = 1e10000
    for i in range(maxit):
        dx = np.diff(Xg2)
        cmin, cmax = np.min(w2*dx), np.max(w2*dx)
        cdiff = cmax - cmin
        # print(i, cmin, cmax, cdiff)
        map[i] = (cdiff, Xg2, X2, w2)
        if cdiff < smallestDiff:
            smallestDiff = cdiff
            smallestI = i
        if cdiff < diffTol:
            if bVerbose:
                print('got result with sufficient accuracy of ', diffTol, 'at iteration', i)
            return Xg2, X2
        Xg2, X2 = fun(Xg2, w2) # iteration
        w2 = interp(X2)
    if bVerbose:
        print('returning best within iterations with diff', smallestDiff, 'at iteration', smallestI)
    return map[smallestI][1:3]


def show_diff(X1, u1, X2, u2, bShow=True):
    plt.plot(X1, u1, '-o')
    plt.plot(X2, u2, '-o')
    if bShow:
        plt.show()


if __name__ == "__main__":
    weightFix = 0.1
    weightPow = 0.5
    x0 = 0.3;c=50
    N = 32
    Xg = np.linspace(0, 1, N + 1)
    print(Xg.shape)
    X = (Xg[1:] + Xg[:-1])/2.
    exact = lambda X: np.cosh(c*(X - x0))**(-2)
    exact_x = lambda X: - 2 * c * exact(X) * np.tanh(c * (X - x0))
    # weights = lambda X: exact(X) + weightFix
    weights = lambda X: np.abs(exact_x(X))**weightPow + weightFix
    from matplotlib import pyplot as plt
    u0 = exact(X)
    w12 = weights(X)
    # for it in range(5, 56, 5):
    it = 50
    print('it=', it)
    Xg2, X2 = iterate_derivation(derive_grid, Xg, weights, maxit=it, bVerbose=True)
    u2 = exact(X2)
    show_diff(X, u0, X2, u2)