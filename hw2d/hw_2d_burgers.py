#!/usr/bin/env python3

import numpy as np
import sys
# integration
from scipy.integrate import ode

# my "package"
sys.path.append('/home/mart/Documents/KybI/2019/python/NewPython2019Oct')
# my stuff
from utils.nonuniform_grid import nonuniform_grid
from hwbasics.HwBasics import Pn_nu, Pnx_nu, Hm_nu
from utils.reprint import reprint
from utils.init_utils import get_X_up_to_power, get_H_and_P

# exact
from utils.burgers_exact import exact_new_mp as exact
# # cluster or not
# from getpass import getuser

# J, nu=1/10, tf=1/2, summax=200, u0i=1, L=1, bHO=False, nua=1, bFindExact=True):
def hw_2d_burgers(Jx, nu, nua=1, bHO=False, bfindExact=True, tf=1/2, u0i=1, L=1, Jy=None, rMax=15, tol=1e-10):
    if Jy is None:
        Jy = Jx
    else:
        print("THIS HAS NOT YET BEEN TESTED THOROUGHLY!")
    Mx = 2**Jx
    Mx2 = 2 * Mx
    My = 2**Jy
    My2 = 2 * My

    # x-part
    Ex, X, X2, X3, Xg = get_X_up_to_power(Jx, nua, 3, bGet0=True)

    Hx, Px1, Px2, Px3, Px4, Px2_1, Px3_1, Px4_1 = get_H_and_P(X, Xg, (1, 2, 3, 4, '2b', '3b', '4b'))

    # reshape X's so they're MATRICES
    Ex = Ex.reshape(1, Mx2)
    X = X.reshape(1, Mx2)
    X2 = X2.reshape(1, Mx2)
    X3 = X3.reshape(1, Mx2)


    # t-part
    Et, T, Tg = get_X_up_to_power(Jy, nua, 1, bGet0=True)
    # sacling
    T = tf * T
    Tg = tf * Tg
    Ht, Pt1 = get_H_and_P(T, Tg, (1,))
    # reshape for matrices
    Et = Et.reshape(My2, 1)
    T = T.reshape(My2, 1)
    Ht = Ht
    Pt1 = Pt1

    # initial condition
    u0 = np.sin(np.pi*X)
    u0x = np.pi*np.cos(np.pi*X)

    # constants
    Rx2 = Px2 - np.dot(Px2_1, X)
    Rx1 = Px1 - np.dot(Px2_1, Ex)
    Dt1 = np.linalg.lstsq(Pt1, Ht)[0].T
    RHSconst = np.dot(Dt1, np.dot(Et, u0))

    # initial guess
    Ur = np.dot(Et, u0)
    Uxr = np.dot(Et, u0x)
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    ax = Axes3D(plt.gcf())
    ax.plot_surface(X, T, Ur)
    plt.show()
    ax = Axes3D(plt.gcf())
    ax.plot_surface(X, T, Uxr)
    plt.show()

    r = 0
    mDiff = 1e10
    while r < rMax and mDiff > tol:
        print('r=%d\tmDiff=%12.9g'%(r, mDiff))
        A = get_A(Mx2, My2, nu, Rx2, Rx1, Hx, Dt1, RHSconst, Ur, Uxr)
        Ucur = np.dot(A, Rx2)
        mDiff = np.max(np.abs(Ucur - Ur))
        Ur = Ucur
        Uxr = np.dot(A, Rx1)
        r += 1
        # from matplotlib import pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # ax = Axes3D(plt.gcf())
        # ax.plot_surface(X, T, Ucur)
        # plt.show()
    return X, T, Ur


def get_A(Nx, Ny, nu, Rx2, Rx1, Hx, Dt1, RHSconst, Ur, Uxr):
    mat = np.zeros((Ny,Nx)*2)
    Ix = np.eye(Nx)
    Iy = np.eye(Ny)

    # import time
    # start = time.time()
    # for i in range(Ny):
    #     for j in range(Nx):
    #         for k in range(Ny):
    #             for l in range(Nx):
    #                 mat[i, j, k, l] = Dt1[i, k] * Rx2[l, j] + Iy[i, k] * Rx2[l, j] * Uxr[i, j] + Ur[i, j] * Iy[i, k] * Rx1[l, j] - nu * Iy[i, k] * Hx[l, j]
    # mat = mat.reshape(Nx*Ny, Nx*Ny, order='F') # Tested to be equivilent to kron product kron(B.T, A)
    # took1 = time.time() - start
    # start = time.time()
    mat =                                                 np.kron(Rx2.T, Dt1) + \
            np.diag(Uxr.reshape(1, Ny*Nx, order='F')[0]) @ np.kron(Rx2.T, Iy) + \
            np.diag( Ur.reshape(1, Ny*Nx, order='F')[0]) @ np.kron(Rx1.T, Iy) - \
                                                      nu * np.kron(Hx.T , Iy)
    # took2 = time.time() - start
    # print('maxdiff',np.max(np.abs(mat-mat2)), 'took1, took2', took1, took2)
    # solvec1 = np.linalg.solve(mat1, f.reshape(My2*Mx2, 1, order='F'))
    # sol1 = solvec1.reshape(My2, Mx2, order='F')

    RHS = Ur * Uxr + RHSconst
    RHS = RHS.reshape(Nx * Ny, 1, order='F') # correct way!
    Uvec = np.linalg.lstsq(mat, RHS)[0]
    return Uvec.reshape(Ny, Nx, order='F') # correct way!


if __name__ == '__main__':
    J = 3
    X, T, U = hw_2d_burgers(J, nu=1/(10*np.pi), Jy=3)
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    ax = Axes3D(plt.gcf())
    ax.plot_surface(X, T, U)
    # plt.pcolormesh(X, T, U)
    # plt.colorbar()
    plt.show()
