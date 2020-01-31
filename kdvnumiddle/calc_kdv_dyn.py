#!/usr/bin/env python3

import numpy as np
import sys
# integration
from scipy.integrate import ode
from scipy.interpolate import interp1d

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

# my "package"
sys.path.append('/home/mart/Documents/KybI/2019/python/hw-burgers')
# my stuff
from utils.nonuniform_grid import nonuniform_grid
from hwbasics.HwBasics import Pn_nu, Pnx_nu, Hm_nu
from utils.reprint import reprint
from utils.utils import plot2D, plot_grid, plot3D

# cluster or not
from getpass import getuser

from dynnu.adaptive_grid import AdaptiveGrid, AdaptiveGridType
    
def solve_kdv(J=3, alpha=1, beta=1, c=1000, tf=1, x0=1/4, fineWidth=3/16, bHO=False, widthTol=0.05, borders=1, a=0.9):
    M2 = 2 * 2 ** J
    # first, get uniform grid
    adaptiveGrid = AdaptiveGrid(J, AdaptiveGridType.DERIV_NU_PLUS_BORDERS, x0, borders, a=a, hw=fineWidth/2)
    X, Xg = nonuniform_grid(J, 1)
    u0x = get_exact_x(X, 0, alpha, beta, c, x0)
    # then 
    Xg, X = adaptiveGrid.get_grid(X, u0x)
    # plot_grid(Xg, X)

    u0 = get_exact(X, 0, alpha, beta, c, x0)

    # H, P
    H_and_P = get_H_and_P(Xg, X, bHO)
    Ps, Pbs = H_and_P[0], H_and_P[1]
    R3, R2, R0 = get_Rs(X, bHO)
    S3, S2, S0 = get_Ss(bHO)
    Sc3 = S3(X, Pbs)
    Sc2 = S2(X, Pbs)
    Sc0 = S0(X, Pbs)
    Dx1, Dx3 = get_Dxs(R3, R2, R0, X, Ps, Pbs)
    # in cluster or not
    cluster = getuser() == "mart.ratas"

    r = ode(fun).set_integrator('lsoda', nsteps=int(1e8))
    r.set_initial_value(u0.flatten(), 0).set_f_params(X, M2, alpha, beta, Dx1, Dx3, Sc3, Sc2, Sc0)
    dt = tf/1e3 # TODO different? adaptive?
    xres = [np.hstack((0, X.flatten(), 1))]
    tres = [0]
    ures = [np.hstack((0, u0.flatten(), 0))]
    ue = [np.hstack((0, u0.flatten(), 0))]
    mdiff = [0]
    xmaxCur = x0
    while r.t < tf:
        r.integrate(r.t+dt)
        toPrint = "%g" % r.t
        if cluster:
            print(toPrint)
        else:
            reprint(toPrint)
        # print('r.y', r.y)
        if np.any(r.y != r.y):
            print("'nan's in the result! CANNOT have this")
            print(r.y)
            break
        # find new (if needed) 
        xres.append(np.hstack((0, X.flatten(), 1)))
        tres.append(r.t)       # TODO - need to get the X axes!
        ures.append(np.hstack((0, r.y, 0))) # reshape?
        ue.append(np.hstack((0, get_exact(X, r.t, alpha, beta, c, x0).flatten(), 0)))
        mdiff.append(np.max(np.abs(ue[-1]-ures[-1])))
        xmaxNew = X[0, np.argmax(r.y)]
        if abs(xmaxNew - xmaxCur) > widthTol:
            Xog, Xo = Xg, X
            Ps_old, Pbs_old = Ps, Pbs
            uxcur = r.y.reshape(1, M2) @ Dx1
            Xg, X = adaptiveGrid.get_grid(Xo, uxcur)
            H_and_P = get_H_and_P(Xg, X, bHO)
            Ps, Pbs = H_and_P[0], H_and_P[1]
            ucur = change_grid(J, r.y, Ps_old, Pbs_old, X, Xog, Xo, R3, bHO)
            Dx1, Dx3 = get_Dxs(R3, R2, R0, X, Ps, Pbs)
            Sc3 = S3(X, Pbs)
            Sc2 = S2(X, Pbs)
            Sc0 = S0(X, Pbs)
            r.set_initial_value(ucur, r.t)
            r.set_f_params(X, M2, alpha, beta, Dx1, Dx3, Sc3, Sc2, Sc0)
            xmaxCur = xmaxNew
    print("DONE:", toPrint)
    T = np.array(tres).reshape(len(tres), 1)
    xx, tt = np.meshgrid(np.hstack((0, X.flatten(), 1)), T)
    xx = np.array(xres)
    print ('TF=', tres[-1])
    U = np.array(ures)
    Ue = np.array(ue)
    print('SHAPES:', U.shape, Ue.shape)
    print('MAX diff:', np.max(mdiff))
    return xx, tt, U, Ue

def get_H_and_P(Xg, X, bHO):
    H = Hm_nu(Xg)
    P2, P3 = [Pn_nu(J, nr, Xg, X.flatten()) for nr in range(2, 4)]
    P2b, P3b = [Pnx_nu(J, nr, 1, Xg) for nr in range(2, 4)]
    if bHO:
        P4, P5 = [Pn_nu(J, nr, Xg, X.flatten()) for nr in range(4, 6)]
        P4b, P5b = [Pnx_nu(J, nr, 1, Xg) for nr in range(4, 6)]
        return (H, None, P2, P3, P4, P5), (None, None, P2b, P3b, P4b, P5b)
    else:
        return (H, None, P2, P3), (None, None, P2b, P3b)

def change_grid(J, ucur, Pso, Pbso, X, Xog, Xo, R3, bHO, bInterpolate=True):
    if not bInterpolate:
        Px2 = np.hstack([Pnx_nu(J, 2, x, Xog) for x in X.flatten()])
        Px3 = np.hstack([Pnx_nu(J, 3, x, Xog) for x in X.flatten()])
        if bHO:
            Px4 = np.hstack([Pnx_nu(J, 4, x, Xog) for x in X.flatten()])
            Px5 = np.hstack([Pnx_nu(J, 5, x, Xog) for x in X.flatten()])
        PsMID = [None, None, Px2, Px3] if not bHO else [None, None, Px2, Px3, Px4, Px5]
        return ucur @ np.linalg.lstsq(R3(Xo, Pso, Pbso), R3(X, PsMID, Pbso))[0]
    else:
        # TODO this only works for homogeneous boundary conditions
        filledUcur = np.hstack((0, ucur.flatten(), 0))
        filledX = np.hstack((0, Xo.flatten(), 1))
        interpolant = interp1d(filledX, filledUcur, 7)
        return interpolant(X.flatten()).reshape(*ucur.shape)

def get_Rs(X, bHO):
    if not bHO:
        c1 = lambda Pbs: 2 * Pbs[3] - 2 * Pbs[2]
        c2 = lambda Pbs: Pbs[2] - 2 * Pbs[3]
        c3 = lambda Pbs: 0 * Pbs[2]
        R3 = lambda X, Ps, Pbs: Ps[3] + c1(Pbs) @ X**2/2 + c2(Pbs) @ X + c3(Pbs)
        R2 = lambda X, Ps, Pbs: Ps[2] + c1(Pbs) @ X      + c2(Pbs)
        R0 = lambda X, Ps, Pbs: Ps[0]
        return R3, R2, R0
    else:
        c1 = lambda Pbs: - Pbs[2]
        c2 = lambda Pbs: 0 * Pbs[2]
        c3 = lambda Pbs: 1/4 * (8 * Pbs[5] - 8 * Pbs[4] + Pbs[2])
        c4 = lambda Pbs: 1/12 * (-24 * Pbs[5] + 12 * Pbs[4] - Pbs[2])
        c5 = lambda Pbs: 0 * Pbs[5]
        R5 = lambda X, Ps, Pbs: Ps[5] + c1(Pbs) @ X**4/24 + c2(Pbs) @ X**3/6 + c3(Pbs) @ X**2/2 + c4(Pbs) @ X + c5(Pbs)
        R4 = lambda X, Ps, Pbs: Ps[4] + c1(Pbs) @ X**3/6  + c2(Pbs) @ X**2/2 + c3(Pbs) @ X      + c4(Pbs)
        R2 = lambda X, Ps, Pbs: Ps[2] + c1(Pbs) @ X       + c2(Pbs)
        return R5, R4, R2

def get_Ss(bHO):
    if not bHO:
        c1 = lambda Pbs: 0
        c2 = lambda Pbs: 0
        c3 = lambda Pbs: 0
        S3 = lambda X, Pbs: c1(Pbs) * X**2/2 + c2(Pbs) * X + c3(Pbs)
        S2 = lambda X, Pbs: c1(Pbs) * X      + c2(Pbs)
        S0 = lambda X, Pbs: 0 * X
        return S3, S2, S0
    else:
        c1 = lambda Pbs: 1
        c2 = lambda Pbs: 0
        c3 = lambda Pbs: - 1/4.
        c4 = lambda Pbs: 1/12.
        c5 = lambda Pbs: 0
        S5 = lambda X, Pbs: c1(Pbs) * X**4/24 + c2(Pbs) * X**3/6 + c3(Pbs) * X**2/2 + c4(Pbs) * X + c5(Pbs)
        S4 = lambda X, Pbs: c1(Pbs) * X**3/6  + c2(Pbs) * X**2/2 + c3(Pbs) * X      + c4(Pbs)
        S2 = lambda X, Pbs: c1(Pbs) * X       + c2(Pbs)
        return S5, S4, S2

def get_Dxs(R3, R2, R0, X, Ps, Pbs):
    Dx1 = np.linalg.lstsq(R3(X, Ps, Pbs), R2(X, Ps, Pbs))[0]
    Dx3 = np.linalg.lstsq(R3(X, Ps, Pbs), R0(X, Ps, Pbs))[0]
    return Dx1, Dx3


# uinfty = 0-> c1 = c2 = c
def get_exact(X, T, alpha, beta, c, x0=1/4):
    a, b = alpha, beta
    return 3 * c * b / a * np.cosh(c**.5/2 * (X - c * b * T - x0))**(-2)

# uinfty = 0 -> c1 = c2 = c (c * beta * t)
def get_exact_x(X, T, alpha, beta, c, x0=1/4):
    a, b = alpha, beta
    return - 3 * c**(3/2.) * b / a * np.tanh(c**.5/2 * (X - c * b * T - x0)) * np.cosh(c**.5/2 * (X - c * b * T - x0))**(-2)


def fun(t, u, X, M2, alpha, beta, Dx1, Dx3, S3, S2, S0):
    u = u.reshape(1, M2)
    ux = (u - S3) @ Dx1 + S2
    uxxx = (u - S3) @ Dx3 + S0
    dudt = - alpha * u * ux - beta * uxxx
    return dudt.flatten()



if __name__ == '__main__':
    # J = 6
    alpha = 6
    beta = .4e-3
    c = .5e4
    x0 = .3
    tf = (1-2*x0)/(beta * c)
    widthTol = 1/25
    fineWidth = .25
    nrOfBorders = 1
    JRange = [5,6]#7]
    # for J in JRange:
    #     mStr = "J=%d, fineWidth = %g"%(J, fineWidth)
    #     print(mStr)
    #     X, T, U, Ue = solve_kdv(J, alpha=alpha, beta=beta, c=c, tf=tf, bHO=False, x0=x0, fineWidth=fineWidth, widthTol=widthTol, borders=nrOfBorders)
        # print(X.shape, T.shape, U.shape, Ue.shape)
        # plot3D(X, T, U, bShow=False, title=mStr),plot3D(X,T,Ue, bShow=False),plot3D(X, T, U-Ue)
    for J in JRange:
        aValues = np.arange(.7, .85, .01)
        if J == 4:
            aValues = [.856,]
        elif J == 5:
            aValues = [.874,]
        elif J == 6:
            aValues = [.940,]
        a = aValues[0]
        maxDiffs = []
        for fineWidth in np.arange(.15, .35, .01):
            for widthTol in np.arange(.01,.1,.005):
                mStr = "J=%d, HOHWM, fineWidth = %g, widthTol=%g"%(J, fineWidth, widthTol)
                print(mStr)
            # for a in aValues:
                try:
                    X, T, U, Ue = solve_kdv(J, alpha=alpha, beta=beta, c=c, tf=tf, bHO=True, x0=x0, fineWidth=fineWidth, widthTol=widthTol, borders=nrOfBorders, a=a)
                except Exception as e:
                    print('\nGOT EXCEPTION\n', e)
                    continue
                md = np.max(np.abs(U - Ue))
                maxDiffs.append((fineWidth, widthTol, np.max(T), md))
            # plot3D(X, T, U, bShow=False, title=mStr),plot3D(X,T,Ue, bShow=False),plot3D(X, T, U-Ue)
        print('MAX diffs:\n', maxDiffs)
        minCur = 1e6
        minFw = maxDiffs[0][0]
        minWt = maxDiffs[0][1]
        for fw, wt, ctf, md in maxDiffs:
            if ctf >= tf and md < minCur:
                minCur = md
                minFw = fw
                minWt = wt
        print('BEST at ', minFw, ",", minWt, " : max diff=", minCur)