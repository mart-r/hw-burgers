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

# exact
from utils.burgers_exact import exact_new_mp as exact


def hw_euler_burgers_newest(J, nu=1/10, tf=1/2, summax=200, u0i=1, L=1, bHO=False, nua=1, bFindExact=True):
    M = 2**J
    M2 = 2 * M

    X, Xg = nonuniform_grid(J, nua)
    X = L * X
    E = X**0
    X2 = (L*X)**2
    X3 = (L*X)**3
    Xg = L * Xg

    if bHO: # even in case of uniform grid - this should (as tested in ML) return the correct result
        P4 = Pn_nu(J, 4, Xg, X)
        P3 = Pn_nu(J, 3, Xg, X)
        P4_1 = Pnx_nu(J, 4, L, Xg).reshape(M2, 1)
        P3_1 = Pnx_nu(J, 3, L, Xg).reshape(M2, 1)

    P2 = Pn_nu(J, 2, Xg, X)
    P1 = Pn_nu(J, 1, Xg, X)
    H  = Hm_nu(Xg)
    P2_1 = Pnx_nu(J, 2, L, Xg).reshape(M2, 1)
    P1_1 = Pnx_nu(J, 1, L, Xg).reshape(M2, 1)

    # reshape X's so they're MATRICES
    E = E.reshape(1, M2)
    X = X.reshape(1, M2)
    X2 = X2.reshape(1, M2)
    X3 = X3.reshape(1, M2)

    u0 = u0i * np.sin(np.pi * X/L)

    if bHO:
        # mat = P4 - P4_1 @ X + 1/6 * P2_1 @ (X - X3)
        # mat1 = P3 - P4_1 + 1/6 * P2_1 @ (1 - 3 * X2)
        # mat2 = P2 - P2_1 @ X
        mat = P4 - np.dot(P4_1, X) + 1/6 * np.dot(P2_1, X - X3)
        mat1 = P3 - P4_1 + 1/6 * np.dot(P2_1, 1 - 3 * X2)
        mat2 = P2 - np.dot(P2_1, X)
    else:
        # mat = P2 - P2_1 @ X
        mat = P2 - np.dot(P2_1, X)
        mat1 = P1 - P2_1
        mat2 = H

    r = ode(fun).set_integrator('lsoda', nsteps=int(1e8))
    r.set_initial_value(u0.flatten(), 0).set_f_params(M2, nu, mat, mat1, mat2)
    dt = 1e-2 # TODO different? adaptive?
    tres = [0]
    ures = [u0.flatten()]
    while r.t < tf:
        r.integrate(r.t+dt)
        reprint("%g" % r.t)
        tres.append(r.t)
        ures.append(r.y) # reshape?
        if (not r.successful and np.max(r.y) > 1e5) or len(tres) > tf/dt: # TODO - this is a "weird" limit
            break
    if bFindExact:
        Ue = get_exact(nu, X, np.array(tres), bHighDPS=abs(nu)<1/50).T
    else:
        Ue = np.zeros((len(tres), M2))
    return X.flatten(), tres, ures, Ue

def get_exact(nu, X, T, bHighDPS=True):
    #exact_new_mp(xv, tv, eps, l=1, u0=1, infty=200):
    infty = 200
    if bHighDPS:
        import mpmath as mp
        mp.mp.dps = 800
        infty = 800
        print('Now calculating exact, this will take some time...')
    return exact(X.flatten(), T, nu, infty=infty)


def fun(t, u, M2, nu, mat0, mat1, mat2):
    u = u.reshape(1, M2) # need to reshape most likely
    # A = (u) / mat0
    # A = np.linalg.solve(mat0.T, u.T).T
    A = np.linalg.lstsq(mat0.T, u.T)[0].T
    # ux = A @ mat1 
    # uxx = A @ mat2
    ux = np.dot(A, mat1)
    uxx = np.dot(A, mat2)
    
    dxdy = - u * ux + nu * uxx
    return dxdy.flatten()


def plot_results(X, T, U, Ue, bShow=True):
    print('Showing results!')
    from matplotlib import pyplot as plt
    i = 0
    mdiff = 0
    for t, u, e in zip(T, U, Ue): # need to reshape e
        if i%100 == 0 and bShow:
            plt.figure()
            plt.plot(X, u.flatten())
            plt.title('t=%f'%t)
            print('title:', 't=%f'%t)
        i += 1
        mdiff = max(mdiff, np.max(u-e))
    if bShow:
        plt.show()
    print('Showed results!, maxDIFF:', mdiff)


def saver(fileName, X, T, U, Ue):
    from utils.utils import save
    save(fileName, X=X, T=T, U=U, Ue=Ue)


if __name__ == '__main__':
    print('starting')
    [X, T, U, Ue] = hw_euler_burgers_newest(2, nu=1/(100 * np.pi), bHO=True, nua=.75)
    plot_results(X, T, U, Ue, bShow=False)
    fileName = "HW_Burgers_test"
    saver(fileName, X, T, U, Ue)