#!/usr/bin/env python3

from scipy.io import savemat
import os
import sys
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def save(filename, **kwargs):
    if not filename.endswith('.mat'):
        filename += '.mat'
    print('saving to "%s/%s":'%(os.getcwd(),filename), '->', kwargs.keys())
    savemat(filename, kwargs)

def plot2D(*args, **kwargs):
    if len(args) > 1:
        X = args[0].flatten()
        Y = args[1].flatten()
    else:
        Y = args[0].flatten()
        X = np.arange(len(Y))
    if "ax" in kwargs:
        ax = kwargs["ax"]
    else:
        plt.figure()
        ax = plt.gca()
    ax.plot(X, Y)
    bShow = True
    if "bShow" in kwargs:
        bShow = bool(kwargs["bShow"])
    if bShow:
        plt.show()

def plot_grid(Xg, X, **kwargs):
    if "ax" in kwargs:
        ax = kwargs["ax"]
    else:
        plt.figure()
        ax = plt.gca()
    ax.plot(Xg.flatten(), Xg.flatten() * 0, 'o')
    ax.plot(X.flatten(), X.flatten() * 0, 'o')
    ax = plt.gca()
    ax.legend(("Grid", "Coll."))
    bShow = True
    if "bShow" in kwargs:
        bShow = bool(kwargs["bShow"])
    if bShow:
        plt.show()

def plot3D(X, T, U, **kwargs):
    X = X.flatten()
    T = T.flatten()
    if "ax" in kwargs:
        ax = kwargs["ax"]
    else:
        ax = Axes3D(plt.figure())
    xx, tt = np.meshgrid(X, T)
    ax.plot_surface(xx, tt, U)
    bShow = True
    if "bShow" in kwargs:
        bShow = bool(kwargs["bShow"])
    if bShow:
        plt.show()