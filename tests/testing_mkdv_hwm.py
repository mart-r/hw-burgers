#!/usr/bin/env python3

import sys, getopt
print (sys.path[0])
sys.path.append('/home/mart/Documents/KybI/2019/python/hw-burgers')
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import numpy as np

from mkdv.mkdv_adaptive_new import get_Ps, get_Pbs, get_R, get_S, get_bc, get_exact

from utils.nonuniform_grid import nonuniform_grid

from utils.utils import plot2D

def test_1(J, X, Xg, la=50., delta=1./50, bDoWM=False, bAexact=False): # lambda and delta values shouldn't really matter
    M2 = 2 * 2**J

    Ps = get_Ps(J, X, Xg)
    Pbs = get_Pbs(J, Xg)
    # print(len(Ps), len(Pbs))
    R3, R2 = get_R(la)[:2]
    S3, S2 = get_S(la, delta)[:2]

    X = X.reshape((1, M2))

    R3c = R3(X, Ps, Pbs)
    R2c = R2(X, Ps, Pbs)
    S3c = S3(X, Pbs)
    S2c = S2(X, Pbs)

    # from wolfram
    if bDoWM:
        R3c_ = Ps[5] + Pbs[5] @ (X**2 * (8 + (-2 + X**2) * la**2)/(-8 + la**2)) \
                + Pbs[4] @ ((X**2 * (1 - X**2) * la**2)/(2 * (-8 + la**2))) \
                - Pbs[2] @ ((X**2 * (1 - X**2))/(3 * (-8 + la**2)))
        S3c_ = -np.sqrt(3./2 * delta) * la * (-8 + la**2 + 2 * X**4 * la**2 - 4 * X**2 * (-4 + la**2))/(-8 + la**2)
        print('(max) diff R3:', np.max(np.abs(R3c - R3c_)))
        print('(max) diff S3:', np.max(np.abs(S3c - S3c_)))

    # print(R3c.shape, R2c.shape, S3c.shape, S2c.shape)

    if bAexact:
        A = np.linalg.lstsq(R3c.T, (get_exact(X, 0, la, delta, x0=3/4.) - S3c).T)[0].T
    else:
        A = np.random.rand(1, M2) * 1e7
    # A = np.array([1,] + [0,]*(M2 - 1)).reshape((1, M2))
    # print(A)

    u = A @ R3c + S3c
    ux = A @ R2c+ S2c

    ax = plot2D(X, u, bShow=False, title='u')
    plot2D(np.linspace(0, .1, 2), -get_bc(la, delta) * np.ones(2), bShow=False, ax=ax)
    plot2D(np.linspace(.9, 1, 2), get_bc(la, delta) * np.ones(2), bShow=False, ax=ax)
    ax = plot2D(X, ux, title='ux', bShow=False)
    plot2D(np.linspace(0, .1, 2), 0 * np.ones(2), ax=ax)

    if bDoWM:
        # try with R3c_ and S3c_
        u = A @ R3c_ + S3c_
        ax = plot2D(X, u, bShow=False, title='u(WM)')
        plot2D(np.linspace(0, .1, 2), -get_bc(la, delta) * np.ones(2), bShow=False, ax=ax)
        plot2D(np.linspace(.9, 1, 2), get_bc(la, delta) * np.ones(2), ax=ax)

    # # check boundaries
    bc = get_bc(la, delta)
    Xbc = np.array(([0, 1])).reshape((1, 2))
    Pbc = [np.hstack((el1, el2)) for el1, el2 in zip(get_Pbs(J, Xg, 0), get_Pbs(J, Xg, 1))]
    ubc = A @ R3(Xbc, Pbc, Pbs) + S3(Xbc, Pbc)
    uxbc = A @ R2(Xbc, Pbc, Pbs) + S2(Xbc, Pbc)
    print("bc", bc, ubc, "diff", ubc[0,0]-(-bc), ubc[-1,-1]-bc, "ux", uxbc[0, 0], '(=diff)')
    # P0s = get_Pbs(J, Xg, 0) # 0
    # uleft = A @ R3(np.array(0), P0s, Pbs)
    # P1s = Pbs
    # uright = A @ R3(np.array(1), P1s, Pbs)
    # print('left:', uleft, 'bc', -bc, 'diff', abs(uleft+bc))
    # print('right:', uright, 'bc', bc, 'diff', abs(uright-bc))

    return u


if __name__ == "__main__":
    J = 7
    X, Xg = nonuniform_grid(J, 1)
    u1 = test_1(J, X, Xg, bAexact=True)
    u2 = test_1(J, X, Xg)
    print('diff:', np.max(np.abs(u1 - u2)))
    plot2D(X, u2, ax=plot2D(X, u1, bShow=False), bShow=False)
    plot2D(X, u1-u2)