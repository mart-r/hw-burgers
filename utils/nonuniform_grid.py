#!/usr/bin/env python3

import numpy as np
import sys
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

def nonuniform_grid(J, a, bOld=False):
    """
    X = nonuniform_grid(J, a, bFixEnd)
    - Get the collocation points (X)
    [X, Xg] = nonuniform_grid(J, a, bFixEnd)
    - Get the collocation points (X) as well as the grid points (Xg)
    Get the nonuniform grid for a certain resolution
    J: resolution
    - number of collocation points M2 = 2^(J+1)
    a: degree of nonuniofrmatiy
    - a < 1 : sparse at the beginning
    - a = 1 : uniform distribution
    - a > 1 : sparse at the end
    bFixEnd: whether or not to fix the last part of the resulting vector
    - in order to try and avoid stacking of grid points

    returns:
    X : collocation points, vector of length M2 = 2^(J+1)
    Xg : grid points, vector of length M2 = 2^(J+1) + 1
    """

    M = 2**J
    M2 = 2 * M
    if a == 1: # UNIFORM
        Xg = np.arange(M2+1)/M2
    else:
        if bOld:
            Xg = (a**(np.arange(M2+1)/M2)-1)/(a-1)
        else:
            Xg = (a**np.arange(M2+1)-1)/(a**M2-1)
    X = (Xg[0:-1] + Xg[1:])/2
    return X, Xg