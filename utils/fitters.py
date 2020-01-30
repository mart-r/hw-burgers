#!/usr/bin/env python3

import sys, getopt
print (sys.path[0])
sys.path.append('/home/mart/Documents/KybI/2019/python/hw-burgers')
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import numpy as np

class NDPolyFitter:

    def __init__(self, vars, value, degs=None, tol=1e-12):
        if len(vars) != 2:
            raise ValueError("Only 2D fitting supported, for now at least")
        if degs is None:
            degs = [2 for el in vars]
        if len(degs) != len(vars):
            raise ValueError("Need the same number of degrees as there are defined variables")
        self.X = vars[0]
        self.Y = vars[1]
        self.degX = degs[0]
        self.degY = degs[1]
        self.Z = value
        self.coefs = polyfit(self.X, self.Y, self.Z, self.degX, self.degY)
        self.coefs[np.abs(self.coefs) < tol] = 0
    
    def __call__(self, x, y):
        res = 0
        ix = 0
        for row in self.coefs:
            iy = 0
            for coef in row:
                res += coef * x**ix * y**iy
                iy += 1
            ix += 1
        return res

def polyfit(X, Y, Z, degX=2, degY=2, doPrint=False):
    X = X.flatten()
    Y = Y.flatten()

    arrays = []
    printOut = []
    for dx in range(0, degX + 1):
        printOut.append([])
        for dy in range(0, degY + 1):
            arrays.append(X**dx * Y**dy)
            if dy == 0 and dx == 0:
                printOut[-1].append('1')
            elif dy == 0:
                printOut[-1].append('X**%d'%dx)
            elif dx == 0:
                printOut[-1].append('Y**%d'%dy)
            else:
                printOut[-1].append('X**%d * Y**%d'%(dx, dy))
    A = np.array(arrays).T
    B = Z.flatten()

    # coeff, r, rank, s = np.linalg.lstsq(A, B)
    coeff = np.linalg.lstsq(A, B)[0].reshape(degX + 1, degY + 1)
    if doPrint:
        for el in printOut:
            print(el)
    return coeff