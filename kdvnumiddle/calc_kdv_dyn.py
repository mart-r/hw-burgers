#!/usr/bin/env python3

import numpy as np
import sys
# integration
from scipy.integrate import ode

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

def get_my_grid(J, lStart, rStart, borders=1, bSmartBorders=True):
    M2 = 2 * 2**J
    if not bSmartBorders:
        Xg = np.linspace(0, lStart, borders + 1)
        Xg = np.hstack((Xg, np.linspace(lStart, rStart, M2 - 2 * borders + 1)[1:-1]))
        Xg = np.hstack((Xg, np.linspace(rStart, 1, borders + 1)))
    else:
        left = lStart 
        right = 1 - rStart
        nrl, nrr = get_smart_grid_left_right(2 * borders, left, right)
        Xg = np.linspace(0, lStart, nrl + 1)
        Xg = np.hstack((Xg, np.linspace(lStart, rStart, M2 - 2 * borders + 1)[1:-1]))
        Xg = np.hstack((Xg, np.linspace(rStart, 1, nrr + 1)))        
    X = (Xg[1:]+Xg[:-1])/2
    return Xg, X.reshape((1, M2))

def get_smart_grid_left_right(totalBorders, left, right):
    nrl = totalBorders * left /(left  + right)
    nrr = totalBorders * right /(left + right)
    nrl, nrr = int(nrl), int(nrr)
    if nrl + nrr < totalBorders:
        if nrl/left < nrr/right:
            nrl, nrr = nrl + 1, nrr
        else:
            nrl, nrr = nrl, nrr + 1
    return nrl, nrr
    
def solve_kdv(J=3, alpha=1, beta=1, c=1000, tf=1, x0=1/4, fineWidth=3/16, bHO=False, widthTol=0.05, borders=1):
    M2 = 2 * 2 ** J
    Xg, X = get_my_grid(J, x0-fineWidth, x0+fineWidth, borders)
    # plot_grid(Xg, X)

    u0 = get_exact(X, 0, alpha, beta, c, x0)

    # H, P
    H_and_P = get_H_and_P(Xg, X, bHO)
    Ps, Pbs = H_and_P[0], H_and_P[1]
    R3, R2, R0 = get_Rs(X, bHO)
    Dx1, Dx3 = get_Dxs(R3, R2, R0, X, Ps, Pbs)
    # in cluster or not
    cluster = getuser() == "mart.ratas"

    r = ode(fun).set_integrator('lsoda', nsteps=int(1e8))
    r.set_initial_value(u0.flatten(), 0).set_f_params(X, M2, alpha, beta, Dx1, Dx3)
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
            Xg, X = get_my_grid(J, xmaxNew-fineWidth, xmaxNew+fineWidth, borders)
            H_and_P = get_H_and_P(Xg, X, bHO)
            Ps, Pbs = H_and_P[0], H_and_P[1]
            ucur = change_grid(J, r.y, Ps_old, Pbs_old, X, Xog, Xo, R3, bHO)
            Dx1, Dx3 = get_Dxs(R3, R2, R0, X, Ps, Pbs)
            r.set_initial_value(ucur, r.t)
            r.set_f_params(X, M2, alpha, beta, Dx1, Dx3)
            xmaxCur = xmaxNew
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

def change_grid(J, ucur, Pso, Pbso, X, Xog, Xo, R3, bHO):
    Px2 = np.hstack([Pnx_nu(J, 2, x, Xog) for x in X.flatten()])
    Px3 = np.hstack([Pnx_nu(J, 3, x, Xog) for x in X.flatten()])
    if bHO:
        Px4 = np.hstack([Pnx_nu(J, 4, x, Xog) for x in X.flatten()])
        Px5 = np.hstack([Pnx_nu(J, 5, x, Xog) for x in X.flatten()])
    PsMID = [None, None, Px2, Px3] if not bHO else [None, None, Px2, Px3, Px4, Px5]
    return ucur @ np.linalg.lstsq(R3(Xo, Pso, Pbso), R3(X, PsMID, Pbso))[0]

def get_Rs(X, bHO):
    if not bHO:
        c1 = lambda Pbs: 2 * Pbs[3] - 2 * Pbs[2]
        c2 = lambda Pbs: Pbs[2] - 2 * Pbs[3]
        c3 = lambda Pbs: 0 * Pbs[2]
        R3 = lambda X, Ps, Pbs: Ps[3] + c1(Pbs) * X**2/2 + c2(Pbs) * X + c3(Pbs)
        R2 = lambda X, Ps, Pbs: Ps[2] + c1(Pbs) * X      + c2(Pbs)
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


def get_Dxs(R3, R2, R0, X, Ps, Pbs):
    Dx1 = np.linalg.lstsq(R3(X, Ps, Pbs), R2(X, Ps, Pbs))[0]
    Dx3 = np.linalg.lstsq(R3(X, Ps, Pbs), R0(X, Ps, Pbs))[0]
    return Dx1, Dx3


# uinfty = 0-> c1 = c2 = c
def get_exact(X, T, alpha, beta, c, x0=1/4):
    a, b = alpha, beta
    return 3 * c * b / a * np.cosh(c**.5/2 * (X - c * b * T - x0))**(-2)


def fun(t, u, X, M2, alpha, beta, Dx1, Dx3):
    u = u.reshape(1, M2)
    ux = u @ Dx1
    uxxx = u @ Dx3
    dudt = - alpha * u * ux - beta * uxxx
    return dudt.flatten()



if __name__ == '__main__':
    # J = 6
    alpha = 6
    beta = .4e-3
    c = .5e4
    x0 = .3
    tf = (1-2*x0)/(beta * c)
    widthTol = 1/10
    fineWidth = 4/16
    JRange = [4, 5, 6]
    for J in JRange:
        mStr = "J=%d, fineWidth = %g"%(J, fineWidth)
        print(mStr)
        X, T, U, Ue = solve_kdv(J, alpha=alpha, beta=beta, c=c, tf=tf, bHO=False, x0=x0, fineWidth=fineWidth, widthTol=widthTol, borders=2)
        print(X.shape, T.shape, U.shape, Ue.shape)
        plot3D(X, T, U, bShow=False, title=mStr),plot3D(X,T,Ue, bShow=False),plot3D(X, T, U-Ue)
    for J in JRange:
        mStr = "J=%d, HOHWM, fineWidth = %g"%(J, fineWidth)
        print(mStr)
        X, T, U, Ue = solve_kdv(J, alpha=alpha, beta=beta, c=c, tf=tf, bHO=True, x0=x0, fineWidth=fineWidth, widthTol=widthTol, borders=2)
        plot3D(X, T, U, bShow=False, title=mStr),plot3D(X,T,Ue, bShow=False),plot3D(X, T, U-Ue)