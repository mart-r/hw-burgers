#!/usr/bin/env python3

import sys
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import numpy as np

# my "package"
sys.path.append('/home/mart/Documents/KybI/2019/python/NewPython2019Oct')
from utils.nonuniform_grid import nonuniform_grid
from hwbasics.HwBasics import Pn_nu, Pnx_nu, Hm_nu


def get_X_up_to_power(J, nua=1, mpow=1, bGet0=False):
    X, Xg = nonuniform_grid(J, nua)
    ret = [X]
    i = 2
    while i <= mpow:
        ret.append(X**i)
        i += 1
    ret += [Xg,]
    if bGet0:
        return [0*X+1,] + ret
    else:
        return ret


def get_H_and_P(X, Xg, use=(1, 2, '2b')):
    H = Hm_nu(Xg)
    Ps = []
    J = int(np.log2(len(X)/2))
    for cur in use:
        nr = 0
        bEnd = isinstance(cur, str)
        if bEnd:
            nr = int(cur[:-1])
        else:
            nr = cur
        if bEnd:
            P = Pnx_nu(J, nr, Xg[-1], Xg).reshape(len(X), 1)
        else:
            P = Pn_nu(J, nr, Xg, X)
        Ps.append(P)

    return [H,] + Ps


if __name__ == '__main__':
    J = 4
    nua = .9
    X, X2, X3, Xg = get_X_up_to_power(J, nua, 3)
    print(X, X2, X3, Xg)

    H, P1, P2, P3, P4, P2_1, P3_1, P4_1 = get_H_and_P(X, Xg, (1,2,3,4, '2b', '3b', '4b'))
    print(H, P2, P3, P4, P2_1, P3_1, P4_1)