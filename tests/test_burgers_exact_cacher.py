#!/usr/bin/env python3

import sys, getopt
sys.path.append('/home/mart/Documents/KybI/2019/python/hw-burgers')
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

from exactcache.burgers_exact_cacher import BurgersExactCacher
from utils.burgers_exact import exact_new_mp as exact
from utils.nonuniform_grid import nonuniform_grid

import numpy as np
import mpmath as mp

def run_tests(nu=1/(100*np.pi)):
    cacher = BurgersExactCacher()
    #x = np.arange(64)[1::2]/64
    x = nonuniform_grid(5, .8)[0]
    t = nonuniform_grid(4, .9)[0]
    cached = cacher.getAt(nu, x, t)
    mp.mp.dps = 800
    calculated = exact(x, t, nu, 1, 1, 800).T
    mdiff = np.max(np.abs(cached - calculated))
    if mdiff > 1e-6:
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        # cached
        ax = Axes3D(plt.gcf())
        xx, tt = np.meshgrid(x,t)
        ax.plot_surface(xx, tt, cached)
        plt.title('cached'),plt.xlabel('x'),plt.ylabel('t')
        # new calc
        ax = Axes3D(plt.figure())
        ax.plot_surface(xx, tt, calculated)
        plt.title('calculated'),plt.xlabel('x'),plt.ylabel('t')
        # DIFF
        ax = Axes3D(plt.figure())
        ax.plot_surface(xx, tt, cached - calculated)
        plt.title('DIFF'),plt.xlabel('x'),plt.ylabel('t')
        # FULL
        ax = Axes3D(plt.figure())
        x, t, U = cacher.getFull(nu)
        xx, tt = np.meshgrid(x, t)
        ax.plot_surface(xx, tt, U)
        plt.title('FULL'),plt.xlabel('x'),plt.ylabel('t')
        plt.show()

        raise ValueError("Expected similar results, got max diff of %g"%mdiff)
    print('Max diff:%g'%mdiff)


if __name__ == '__main__':
    run_tests()