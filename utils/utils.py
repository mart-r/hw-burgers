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
    pType = "-"
    if "type" in kwargs:
        pType = kwargs["type"]
    ax.plot(X, Y, pType)
    bShow = True
    if "bShow" in kwargs:
        bShow = bool(kwargs["bShow"])
    if bShow:
        plt.show()
    return ax

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
    if "title" in kwargs:
        ax.set_title(kwargs["title"])
    bShow = True
    if "bShow" in kwargs:
        bShow = bool(kwargs["bShow"])
    if bShow:
        plt.show()

def plot3D(X, T=None, U=None, **kwargs):
    if T is None and U is None:
        U = X
        X = np.arange(U.shape[1])
        T = np.arange(U.shape[0])
    if X.shape != T.shape:
        X = X.flatten()
        T = T.flatten()
        X, T = np.meshgrid(X, T)
    if "ax" in kwargs:
        ax = kwargs["ax"]
    else:
        ax = Axes3D(plt.figure())
    ax.plot_surface(X, T, U)
    if "title" in kwargs:
        ax.set_title(kwargs["title"])
    bShow = True
    if "bShow" in kwargs:
        bShow = bool(kwargs["bShow"])
    if bShow:
        plt.show()