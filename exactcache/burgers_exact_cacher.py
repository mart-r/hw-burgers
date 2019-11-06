#!/usr/bin/env python3

import numpy as np
from scipy.io import savemat, loadmat
from scipy.interpolate import interp2d
import sys, os
import mpmath as mp
# integration

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

# my "package"
sys.path.append('/home/mart/Documents/KybI/2019/python/hw-burgers')

# exact
from utils.burgers_exact import exact_new_mp as exact

class BurgersExactCacher:
    __cacheFile = 'burgers_exact_cache.mat'
    __cache = {}

    def __init__(self, Nx=1000, Nt=1000, tf=0.5):
        self.tf = tf
        self.dx = 1/Nx
        self.dt = self.tf/Nt
        if os.path.isfile(self.__cacheFile):
            inD = loadmat(self.__cacheFile)
            typesSaved = inD['types']
            for nu, Xv, Tv in typesSaved:
                nu = nu[0,0]
                print (nu, Xv.shape, Tv.shape, inD[str(nu)].shape)
                try:
                    self.__cache[nu] = BurgersExactAtNu(Xv.flatten(), Tv.flatten(), inD[str(nu)])
                except ValueError as e:
                    print('Unable to use nu=%f'%nu)
                    print(e)
    
    def __calc_exact(self, nu):
        X = np.arange(0, 1, self.dx)
        T = np.arange(0, self.tf, self.dt) # TODO - calculate tf from nu
        infty = 200
        bHighDPS = nu < 0.01
        if bHighDPS:
            oldDps = mp.mp.dps
            mp.mp.dps = 800 
            infty = 800 
        U = exact(X, T, nu, infty=infty)
        if bHighDPS:
            mp.mp.dps = oldDps
        return BurgersExactAtNu(X, T, U)

    def __add_exact(self, nu, curExact):
        self.__cache[nu] = curExact
        self.__save()
    
    def __save(self):
        outDict = {}
        types = []
        for nu in self.__cache.keys():
            myExact = self.__cache[nu]
            types.append((nu, myExact.getX(), myExact.getT()))
            outDict[str(nu)] = myExact.getU()
        outDict['types'] = types
        savemat(self.__cacheFile, outDict)
    
    def __getFor(self, nu):
        if nu in self.__cache:
            return self.__cache[nu]
        else:
            curExact = self.__calc_exact(nu)
            print('Calculated exact...', curExact.getU().shape)
            self.__add_exact(nu, curExact)
            return curExact
            
    def getAt(self, nu, Xv, Tv):
        curExact = self.__getFor(nu)
        interpolant = interp2d(curExact.getX(), curExact.getT(), curExact.getU(), 'cubic')
        return interpolant(Xv, Tv).T

    def getFull(self, nu):
        curExact = self.__getFor(nu)
        return curExact.getX(), curExact.getT(), curExact.getU()


class BurgersExactAtNu:    
    def __init__(self, X, T, U):
        if X is None:
            raise ValueError("X of an exact solution cannot be None")
        if T is None:
            raise ValueError("T of an exact solution cannot be None")
        if U is None:
            raise ValueError("U of an exact solution cannot be None")
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array!")
        if not isinstance(T, np.ndarray):
            raise ValueError("T must be a numpy array!")
        if not isinstance(U, np.ndarray):
            raise ValueError("U must be a numpy array!")
        nx = max(X.shape)
        nt = max(T.shape)
        m, n = U.shape
        if (nx != m or nt != n):
            raise ValueError("Input mis-shaped!lenX:%d lenT:%d, U:(%d,%d)"%(nx, nt, m, n))
        self.__X = X
        self.__T = T
        self.__U = U

    def getX(self):
        return self.__X

    def getT(self):
        return self.__T

    def getU(self):
        return self.__U

if __name__ == '__main__':
    cacher = BurgersExactCacher()
    nu = 1/(100 * np.pi)
    xv = np.arange(16)[1::2]/16
    tv = xv * 0.5
    U = cacher.getAt(nu, xv, tv)
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    ax = Axes3D(plt.gcf())
    print(xv.shape, tv.shape, U.shape)
    xx, tt = np.meshgrid(xv, tv)
    ax.plot_wireframe(xx, tt, U)
    plt.show()