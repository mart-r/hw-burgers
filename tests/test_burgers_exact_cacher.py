#!/usr/bin/env python3

import sys, getopt
sys.path.append('/home/mart/Documents/KybI/2019/python/hw-burgers')
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

from exactcache.burgers_exact_cacher import BurgersExactCacher
from utils.burgers_exact import exact_new_mp as exact

import numpy as np
import mpmath as mp

def run_tests(nu=1/(100*np.pi)):
    cacher = BurgersExactCacher()
    x = np.arange(16)[1::2]/16
    t = x/2
    cached = cacher.getAt(nu, x, t)
    mp.mp.dps = 800
    calculated = exact(x, t, nu, 1, 1, 800).T
    mdiff = np.max(np.abs(cached - calculated))
    if mdiff > 1e-8:
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        ax = Axes3D(plt.gcf())
        xx, tt = np.meshgrid(x,t)
        ax.plot_surface(xx, tt, cached)
        plt.title('cached'),plt.xlabel('x'),plt.ylabel('t')
        ax = Axes3D(plt.figure())
        ax.plot_surface(xx, tt, calculated)
        plt.title('calculated'),plt.xlabel('x'),plt.ylabel('t')
        # FULL
        ax = Axes3D(plt.figure())
        x, t, U = cacher.getFull(nu)
        xx, tt = np.meshgrid(x, t)
        ax.plot_surface(xx, tt, U)
        plt.title('FULL'),plt.xlabel('x'),plt.ylabel('t')
        plt.show()

        raise ValueError("Expected similar results, got max diff of %f"%mdiff)
    print('Max diff:%g'%mdiff)


if __name__ == '__main__':
    run_tests()