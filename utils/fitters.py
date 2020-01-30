#!/usr/bin/env python3

import sys, getopt
print (sys.path[0])
sys.path.append('/home/mart/Documents/KybI/2019/python/hw-burgers')
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import numpy as np

class NDPolyFitter:

    def __init__(self, variables, value, degs=None, tol=1e-12):
        if degs is None:
            degs = [2 for el in variables]
        if len(degs) != len(variables):
            raise ValueError("Need the same number of degrees as there are defined variables")
        self.vars = variables
        self.degs = degs
        self.Z = value
        self.coefs = ndpolyfit(self.vars, self.Z, self.degs)
        self.coefs[np.abs(self.coefs) < tol] = 0
    
    def __helper(self, variables, coefs, degs):
        if len(variables) > 2:
            res = 0
            maxDeg = degs[0]
            var = variables[0]
            for d in range(maxDeg + 1):
                res += var**d * self.__helper(variables[1:], coefs[d], degs[1:])
        else:
            res = 0
            i1 = 0
            for row in coefs:
                i2 = 0
                for coef in row:
                    res += coef * variables[0]**i1 * variables[1]**i2
                    i2 += 1
                i1 += 1
        return np.squeeze(np.array(res))
    
    def __call__(self, *variables):
        res = self.__helper(variables, self.coefs, self.degs)
        return res

def ndpolyfit(variables, val, degs=None):
    if degs is None:
        degs = [3 for el in variables]
    if len(variables) != len(degs):
        raise ValueError("Need the same amount of variables and degrees")
    variables = [v.flatten() for v in variables]
    arrays = __helper(variables, degs)
    size = (np.prod(np.shape(variables[0])), np.prod([d + 1 for d in degs]))
    A = np.array(arrays).T.reshape(size, order='F')
    B = val.flatten()
    res = np.linalg.lstsq(A, B)[0]
    return res.reshape([d + 1 for d in degs])

def __helper(var, deg):
    arrays = []
    if len(var) > 2:
        cdeg = deg[0]
        cvar = var[0]
        for d in range(0, cdeg + 1):
            arrays.append(cvar**d * __helper(var[1:], deg[1:]))
    else:
        arrays.append(__helper0(*var, *deg))
    res = np.squeeze(np.array(arrays))
    return res

def __helper0(X, Y, degX, degY, returnPrint=False):
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
    if returnPrint:
        return arrays, printOut
    else:
        return arrays

def polyfit(X, Y, Z, degX=2, degY=2, doPrint=False):
    X = X.flatten()
    Y = Y.flatten()

    arrays, printOut = __helper0(X, Y, degX, degY, True)
    A = np.array(arrays).T
    B = Z.flatten()

    # coeff, r, rank, s = np.linalg.lstsq(A, B)
    coeff = np.linalg.lstsq(A, B)[0].reshape(degX + 1, degY + 1)
    if doPrint:
        for el in printOut:
            print(el)
    return coeff