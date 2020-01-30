#!/usr/bin/env python3

import sys, getopt
print (sys.path[0])
sys.path.append('/home/mart/Documents/KybI/2019/python/hw-burgers')
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import numpy as np

from utils.fitters import polyfit, NDPolyFitter


if __name__ == "__main__":
    
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    X, Y = np.meshgrid(x, y, copy=False)
    # Z = 1 + 2 * Y + 3 * Y**2 + 4 * X + 5 * X * Y + 6 * X * Y**2 + 7 * X**2 + 8 * X**2 * Y + 9 * x**2 * Y**2
    Z = 2 * X**2 + Y**2
    coefs = polyfit(X, Y, Z, degX=3, degY=3, doPrint=True)
    print('coeff:', coefs)

    print("with class")
    p = NDPolyFitter((X, Y), Z)
    print('Fitter:', p)
    print('fitter coefs:', p.coefs)
    print('DIFF:', p(X,Y) - Z)