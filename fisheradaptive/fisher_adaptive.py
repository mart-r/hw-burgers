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
    
def solve_FE(J=3, eqa=10, eqb=1/500, x0=1/4, fineWidth=3/16, bHO=True, widthTol=1/25, borders=1, a=0.9, bc=[1, 0] ):
    # calculate tf
    v = 5 * np.sqrt(eqa * eqb/6)
    tf = (1 - 2 * x0)/v

    M2 = 2 * 2 ** J
    adaptiveGrid = AdaptiveGrid(J, AdaptiveGridType.DERIV_NU_PLUS_BORDERS, x0, borders, 
                    a=a, hw=fineWidth, onlyMin=True, bc=bc, Nper=int(M2/2)-borders)
    # first, get uniform grid
    X, Xg = nonuniform_grid(J, 1)
    u0x = get_exact_x(X, 0, eqa, eqb, x0)
    # then 
    Xg, X = adaptiveGrid.get_grid(X, u0x)
    # plot_grid(Xg, X)

    #my COEF (for EXACT middle)
    c1 = np.sqrt(eqa/(6 * eqb))
    myCoef = 1/c1 * np.log(.5)

    u0 = get_exact(X, 0, eqa, eqb, x0)

    # H, P
    H_and_P = get_H_and_P(Xg, X, bHO)
    Ps, Pbs = H_and_P[0], H_and_P[1]
    R2, R1, R0 = get_Rs(X, bHO)
    Dx = get_Dxs(R2, R0, X, Ps, Pbs)
    # in cluster or not
    cluster = getuser() == "mart.ratas"

    r = ode(fun).set_integrator('lsoda', nsteps=int(1e8))
    r.set_initial_value(u0.flatten(), 0).set_f_params(X, M2, eqa, eqb, Dx)
    dt = tf/1e3 # TODO different? adaptive?
    xres = [np.hstack((0, X.flatten(), 1))]
    tres = [0]
    ures = [np.hstack((bc[0], u0.flatten(), bc[1]))]
    ue = [np.hstack((bc[0], u0.flatten(), bc[1]))]
    mdiff = [0]
    xmaxCur = x0
    while r.t < tf:
        r.integrate(r.t+dt)
        toPrint = "tf=%g" % r.t
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
        ures.append(np.hstack((bc[0], r.y, bc[1]))) # reshape?
        ue.append(np.hstack((bc[0], get_exact(X, r.t, eqa, eqb, x0).flatten(), bc[1])))
        mdiff.append(np.max(np.abs(ue[-1]-ures[-1])))
        # plot2D(X, r.y, bShow=False)
        uxcur = r.y.reshape(1, M2) @ (np.linalg.lstsq(R2(X, Ps, Pbs), R1(X, Ps, Pbs))[0])
        xmaxNew = get_cur_mid(X, uxcur) # TODO - finding middle is not trivial...
        # realMid = X[0, np.argmin(get_exact_x(X, r.t, eqa, eqb, x0))] + myCoef
        # if len(tres)%10 == 0:
        #     print(xmaxCur, xmaxNew, realMid)
        # xmaxNew = realMid
        # plot2D(X, uxcur)
        if abs(xmaxNew - xmaxCur) > widthTol:
            Xog, Xo = Xg, X
            Ps_old, Pbs_old = Ps, Pbs
            Xg, X = adaptiveGrid.get_grid(Xo, uxcur)
            H_and_P = get_H_and_P(Xg, X, bHO)
            Ps, Pbs = H_and_P[0], H_and_P[1]
            # plot_grid(Xg, X)
            try:
                ucur = change_grid(J, r.y, Ps_old, Pbs_old, X, Xog, Xo, R2, bHO, bc=bc)
            except ValueError as e:
                print(xmaxNew)
                print(Xo, Xog, "\n", X, Xg)
                print(e)
                break
            Dx = get_Dxs(R2, R0, X, Ps, Pbs)
            # plot2D(Xo, r.y,bShow=False),plot2D(X, ucur) # TODO - remove
            r.set_initial_value(ucur, r.t)
            r.set_f_params(X, M2, alpha, beta, Dx)
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

def get_cur_mid(X, ux, target=0.5):
    # iclosest = np.argmin(np.abs(ux - target))
    # x1 = X[0, iclosest]
    # # print(x1, iclosest)
    # val1 = ux[iclosest]
    # if (val1 > target):
    #     x2 = X[0, iclosest + 1]
    #     val2 = ux[iclosest + 1]
    # else:
    #     x2 = X[0, iclosest - 1]
    #     val2 = ux[iclosest - 1]
    # diff1 = abs(val1 - target)
    # diff2 = abs(val2 - target)
    # return (diff1 * x2 + diff2 * x1)/(diff1 + diff2)
    return X[0, np.argmin(ux)]


def get_H_and_P(Xg, X, bHO):
    H = Hm_nu(Xg)
    P2 = Pn_nu(J, 2, Xg, X.flatten())
    P2b = Pnx_nu(J, 2, 1, Xg) 
    P1 = Pn_nu(J, 1, Xg, X.flatten())
    if bHO:
        P4 = Pn_nu(J, 4, Xg, X.flatten())
        P4b = Pnx_nu(J, 4, 1, Xg)
        P3 = Pn_nu(J, 3, Xg, X.flatten())
        return (H, P1, P2, P3, P4), (None, None, P2b, None, P4b)
    else:
        return (H, P1, P2)        , (None, None, P2b)

def change_grid(J, ucur, Pso, Pbso, X, Xog, Xo, R2, bHO, bInterpolate=False, bc=None):
    if not bInterpolate:
        Px2 = np.hstack([Pnx_nu(J, 2, x, Xog) for x in X.flatten()])
        if bHO:
            Px4 = np.hstack([Pnx_nu(J, 4, x, Xog) for x in X.flatten()])
        PsMID = [None, None, Px2] if not bHO else [None, None, Px2, None, Px4]
        return ucur @ np.linalg.lstsq(R2(Xo, Pso, Pbso), R2(X, PsMID, Pbso))[0]
    else:
        filledUcur = np.hstack((bc[0], ucur.flatten(), bc[1]))
        filledX = np.hstack((0, Xo.flatten(), 1))
        interpolant = interp1d(filledX, filledUcur)
        unew = interpolant(X.flatten())
        return unew.reshape(*ucur.shape)

def get_Rs(X, bHO):
    if not bHO:
        c1 = lambda Pbs: - Pbs[2] - 1
        c2 = lambda Pbs: 0 * Pbs[2] + 1
        R2 = lambda X, Ps, Pbs: Ps[2] + c1(Pbs) * X      + c2(Pbs)
        R1 = lambda X, Ps, Pbs: Ps[1] + c1(Pbs) 
        R0 = lambda X, Ps, Pbs: Ps[0]
        return R2, R1, R0
    else:
        c1 = lambda Pbs: - Pbs[2]
        c2 = lambda Pbs: 0 * Pbs[2]
        c3 = lambda Pbs: 1/6 * (-6 * Pbs[4] + Pbs[2] - 6)
        c4 = lambda Pbs: 0 * Pbs[2] + 1
        R4 = lambda X, Ps, Pbs: Ps[4] + c1(Pbs) @ X**3/6  + c2(Pbs) @ X**2/2 + c3(Pbs) @ X      + c4(Pbs)
        R3 = lambda X, Ps, Pbs: Ps[3] + c1(Pbs) @ X**2/2  + c2(Pbs) @ X      + c3(Pbs)
        R2 = lambda X, Ps, Pbs: Ps[2] + c1(Pbs) @ X       + c2(Pbs)
        return R4, R3, R2


def get_Dxs(R2, R0, X, Ps, Pbs):
    Dx = np.linalg.lstsq(R2(X, Ps, Pbs), R0(X, Ps, Pbs))[0]
    return Dx


def get_exact(X, T, a, b, x0=1/4):
    c1 = np.sqrt(a / (6 * b))
    c2 = 5 * np.sqrt(a * b / 6)
    return (1 + np.exp(c1 * (X - c2 * T - x0)))**(-2)


def get_exact_x(X, T, a, b, x0=1/4):
    c1 = np.sqrt(a / (6 * b))
    c2 = 5 * np.sqrt(a * b / 6)
    return - 2 * c1 * np.exp(c1 * (X - c2 * T - x0)) / (1 + np.exp(c1 * (X - c2 * T - x0)))**3


def fun(t, u, X, M2, a, b, Dx):
    u = u.reshape(1, M2)
    uxx = u @ Dx
    dudt = b * uxx + a * u * (1 - u)
    return dudt.flatten()



if __name__ == '__main__':
    # J = 6
    alpha = 6
    beta = .4e-3
    c = .5e4
    x0 = .3
    tf = (1-2*x0)/(beta * c)
    widthTol = 1/25
    fineWidth = .2
    nrOfBorders = 2
    JRange = [4,5,6]#7]
    for J in JRange:
        mStr = "J=%d, HOHWM, fineWidth = %g"%(J, fineWidth)
        print(mStr)
        aValues = np.arange(.7, .85, .01)
        if J == 4:
            aValues = [.999,]
        elif J == 5:
            aValues = [.999,]
        elif J == 6:
            aValues = [.999,]
        for a in aValues:
            # X, T, U, Ue = solve_kdv(J, alpha=alpha, beta=beta, c=c, tf=tf, bHO=True, x0=x0, fineWidth=fineWidth, widthTol=widthTol, borders=nrOfBorders, a=a)
            X, T, U, Ue = solve_FE(J, x0=x0, a=a, widthTol=widthTol,fineWidth=fineWidth)
            plot3D(X, T, U, bShow=False, title=mStr, zlims=[0, 1]),plot3D(X,T,Ue, bShow=False),plot3D(X, T, U-Ue)