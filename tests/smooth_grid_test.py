#!/usr/bin/env python3

import sys, getopt
print (sys.path[0])
sys.path.append('/home/mart/Documents/KybI/2019/python/hw-burgers')
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")
import numpy as np
from kdvnumiddle.calc_kdv_dyn import get_my_smooth_borders
from utils.utils import plot_grid

def do_test(x0=0.5, w=4/16, N=32):
    print(x0)
    Xg = get_my_smooth_borders(N, x0 - w, x0 + w, 2)
    print(Xg)

    X = (Xg[1:]+Xg[:-1])/2

    plot_grid(Xg, X, bShow=False)
    from matplotlib import pyplot as plt
    plt.plot(x0, .1, 'o')
    plt.figure()
    plt.plot(np.arange(N + 1)/N, Xg, 'o'),plt.plot((np.arange(N) + 0.5)/N, X, 'o')
    plt.plot(x0, x0, 'o')
    plt.show()



if __name__ == '__main__':
    for gap in np.arange(.3, .71, .2):
        do_test(gap)