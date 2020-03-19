#!/usr/bin/env python3

import sys, getopt
print (sys.path[0])
sys.path.append('/home/mart/Documents/KybI/2019/python/hw-burgers')
sys.path.append('/home/mart/Documents/KybI/2019/python/hw-burgers/utils')
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import numpy as np

from utils.utils import plot_grid, plot2D

def get_adaptive_grid(l, M, mid, fw, nb, nbl, nbr):
    if nbl + nbr != 2 * nb:
        raise ValueError("%d + %d != 2 * %d"%(nbl, nbr, nb))
    if l <= nbl:
        return l * float(mid - fw) / nbl
    if nbl < l and l < 2 * M - nbr:
        return (l - nbl) * 2 * fw / (2 * M - 2 * nb) + (mid - fw)
    if l >= 2 * M - nbr:
        return (l - (2 * M - nbr)) * (1 - (mid + fw))/nbr + (mid + fw)


if __name__ == "__main__":
    M = 8
    Xg = np.array([get_adaptive_grid(l, M, 0.5, 0.2, 1, 1, 1) for l in range(0, 2 * M + 1)])
    X = (Xg[:-1]+ Xg[1:])/2.
    print(len(Xg), Xg)
    plot_grid(Xg, X)
    plot2D(np.diff(Xg))
