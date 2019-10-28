#!/usr/bin/env python3

import numpy as np
import sys
# integration
from scipy.integrate import ode

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

# my "package"
sys.path.append('/home/mart/Documents/KybI/2019/python/NewPython2019Oct')
# my stuff
from utils.nonuniform_grid import nonuniform_grid
from hwbasics.HwBasics import Pn_nu, Pnx_nu, Hm_nu
from utils.reprint import reprint

# exact
from utils.burgers_exact import exact_new_mp as exact
# cluster or not
from getpass import getuser


def hw_euler_burgers_newest(J, nu=1/10, tf=1/2, summax=200, u0i=1, L=1, bHO=False, hos=1, nua=1, bFindExact=True):
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
        if hos >= 2:
            P6 = Pn_nu(J, 6, Xg, X)
            P5 = Pn_nu(J, 5, Xg, X)
            P6_1 = Pnx_nu(J, 6, L, Xg).reshape(M2, 1)
            P5_1 = Pnx_nu(J, 5, L, Xg).reshape(M2, 1)
        if hos >= 3:
            P8 = Pn_nu(J, 8, Xg, X)
            P7 = Pn_nu(J, 7, Xg, X)
            P8_1 = Pnx_nu(J, 8, L, Xg).reshape(M2, 1)
            P7_1 = Pnx_nu(J, 7, L, Xg).reshape(M2, 1)
        elif hos >= 4:
            raise Exception("Not implemented!")

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
        if hos == 1:
            mat = P4 - np.dot(P4_1, X) + 1/6 * np.dot(P2_1, X - X3)
            mat1 = P3 - P4_1 + 1/6 * np.dot(P2_1, 1 - 3 * X2)
            mat2 = P2 - np.dot(P2_1, X)
        elif hos == 2:
            print('\n\n\ns=2\n\n\n')
            c1 =  - P2_1
            c2 = P1_1 * 0
            c3 = 1/6 * (-6 * P4_1 + P2_1)
            c4 = P1_1 * 0
            c5 = 1/360 * (-360 * P6_1 + 60 * P4_1 - 7 * P2_1)
            c6 = P1_1 * 0
            mat  = P6 + c1 @ (X**5)/120 + c2 @ (X**4)/24 + c3 @ (X**3)/6 + c4 @ (X**2)/2 + c5 @ X + c6 @ E
            mat1 = P5 + c1 @ (X**4)/24  + c2 @ (X**3)/6  + c3 @ (X**2)/2 + c4 @ X        + c5 @ E
            mat2 = P4 + c1 @ (X**3)/6   + c2 @ (X**2)/2  + c3 @ X        + c4 @ E
        elif hos == 3:
            print('\n\n\ns=3\n\n\n')
            c1 =  - P2_1
            c2 = P1_1 * 0
            c3 = 1/6 * (-6 * P4_1 + P2_1)
            c4 = P1_1 * 0
            c5 = 1/360 * (-360 * P6_1 + 60 * P4_1 - 7 * P2_1)
            c6 = P1_1 * 0
            c7 = 1/15120 * (-15120 * P8_1 + 2520 * P6_1 - 294 * P4_1 + 31 * P2_1)
            c8 = P1_1 * 0
            mat  = P8 + c1 @ (X**7)/5040 + c2 @ (X**6)/720 + c3 @ (X**5)/120 + c4 @ (X**4)/24 + c5 @ (X**3)/6 + c6 @ (X**2)/2 + c7 @ X + c8 @ E
            mat1 = P7 + c1 @ (X**6)/720  + c2 @ (X**5)/120 + c3 @ (X**4)/24  + c4 @ (X**3)/6  + c5 @ (X**2)/2 + c6 @ X        + c7 @ E
            mat2 = P6 + c1 @ (X**5)/120  + c2 @ (X**4)/24  + c3 @ (X**3)/6   + c4 @ (X**2)/2  + c5 @ X        + c6 @ E
        else:
            raise Exception("Not implemented!")
    else:
        # mat = P2 - P2_1 @ X
        mat = P2 - np.dot(P2_1, X)
        mat1 = P1 - np.dot(P2_1, E)
        mat2 = H

    # in cluster or not
    cluster = getuser() == "mart.ratas"

    r = ode(fun).set_integrator('lsoda', nsteps=int(1e8))
    r.set_initial_value(u0.flatten(), 0).set_f_params(M2, nu, mat, mat1, mat2)
    dt = 1e-2 # TODO different? adaptive?
    tres = [0]
    ures = [u0.flatten()]
    while r.t < tf:
        r.integrate(r.t+dt)
        toPrint = "%g" % r.t
        if cluster:
            print(toPrint)
        else:
            reprint(toPrint)
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
    [X, T, U, Ue] = hw_euler_burgers_newest(4, nu=1/(100 * np.pi), bHO=True, hos=2, nua=.8)
    plot_results(X, T, U, Ue, bShow=False)
    fileName = "HW_Burgers_test"
    saver(fileName, X, T, U, Ue)