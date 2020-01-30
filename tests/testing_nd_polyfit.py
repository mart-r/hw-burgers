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
    diff = p(X,Y) - Z
    # print('DIFF:', diff)
    print('MAX difF:', np.max(np.abs(diff)))

    print("Testing 3D")
    z = np.linspace(0, 1, 5)
    XX, YY, ZZ = np.meshgrid(x, y, z)
    print('shapes of input:', XX.shape, YY.shape, ZZ.shape)

    VAL = XX**2 - YY**2 + 2 * ZZ
    print('shape of data:', VAL.shape)

    p2 = NDPolyFitter((XX, YY, ZZ), VAL)
    print("Got fitter:", p2)
    print("With coefficients:", p2.coefs)
    diff = p2(XX, YY, ZZ) - VAL
    # print("And diff:", diff)
    print('MAX difF:', np.max(np.abs(diff)))

    print("Testing 4D")
    z2 = np.linspace(0, 1, 5)
    XX, YY, ZZ, Z2Z2 = np.meshgrid(x, y, z, z2)
    print('shapes of input:', XX.shape, YY.shape, ZZ.shape, Z2Z2.shape)

    VAL = XX**2 - YY**2 + 2 * ZZ + 6 * Z2Z2
    print('shape of data:', VAL.shape)

    p2 = NDPolyFitter((XX, YY, ZZ, Z2Z2), VAL, tol=1e-10)
    print("Got fitter:", p2)
    print("With coefficients:", p2.coefs)
    diff = p2(XX, YY, ZZ, Z2Z2) - VAL
    # print("And diff:", diff)
    print('MAX difF:', np.max(np.abs(diff)))