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
    
def solve_SG(J=3, c=1-1/1e4, x0=1/4, fineWidth=3/16, bHO=True, widthTol=1/25, borders=1, a=0.9, bc=[0, 2 * np.pi], bDebug=True):
    # calculate tf
    tf = (1 - 2 * x0)/c

    M2 = 2 * 2 ** J
    adaptiveGrid = AdaptiveGrid(J, AdaptiveGridType.DERIV_NU_PLUS_BORDERS, x0, borders, 
                    a=a, hw=fineWidth, onlyMax=True, bc=bc, Nper=int(M2/2)-borders)
    # adaptiveGrid = AdaptiveGrid(J, AdaptiveGridType.STATIONARY_MIDDLE, x0, borders, gap=.11)
    # first, get uniform grid
    X, Xg = nonuniform_grid(J, 1)
    u0x = get_exact_x(X, 0, c, x0)
    # then 
    Xg, X = adaptiveGrid.get_grid(X, u0x)
    # plot_grid(Xg, X)

    u0 = get_exact(X, 0, c, x0)
    v0 = get_exact_t(X, 0, c, x0)
    # ax = plot2D(X, u0, bShow=False)
    # plot2D(X, v0, ax=ax, legend=("u0", v0))
    print('How well are boundary conditions enforced? left:', bc[0]-u0[0,0], 'right:', bc[1]-u0[0,-1])

    # H, P
    H_and_P = get_H_and_P(Xg, X, bHO)
    Ps, Pbs = H_and_P[0], H_and_P[1]
    R2, R1, R0 = get_Rs(X, bHO)
    S2, S1, S0 = get_Ss(X, bHO) 
    Dx = get_Dxs(R2, R1, R0, X, Ps, Pbs)
    Sc2 = S2(X, Pbs)
    Sc1 = S1(X, Pbs)
    Sc0 = S0(X, Pbs)
    # in cluster or not
    cluster = getuser() == "mart.ratas"

    r = ode(fun).set_integrator('lsoda', nsteps=int(1e9))
    r.set_initial_value(np.hstack((u0.flatten(), v0.flatten())), 0).set_f_params(X, M2, Dx, Sc2, Sc1, Sc0)
    dt = 1e-3#tf/1e3 # TODO different? adaptive?
    xres = [np.hstack((0, X.flatten(), 1))]
    tres = [0]
    ures = [np.hstack((bc[0], u0.flatten(), bc[1]))]
    ue = [np.hstack((bc[0], u0.flatten(), bc[1]))]
    mdiff = [0]
    xmaxCur = x0
    mids = [x0,]
    rmids1 = [get_cur_mid(X, get_exact(X, 0, c, x0)),]
    exact_mid = lambda t: c * t + x0
    rmids2 = [exact_mid(0),]
    swaps = []
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
        ucur = r.y[:M2]
        vcur = r.y[M2:]
        # if len(mids) < 10:
        #     vstart = vcur[0]
        #     vmax = np.max(np.abs(vcur))
        #     vstartRel = vstart/vmax
        #     toPrint = "%s : %g, %g"%(toPrint, vstart, vstartRel)
        #     plot2D(X, vcur, ax=plot2D(X, ucur, bShow=False), legend=("u", "v"), title=toPrint)
        ures.append(np.hstack((bc[0], ucur, bc[1]))) # reshape?
        ue.append(np.hstack((bc[0], get_exact(X, r.t, c, x0).flatten(), bc[1])))
        mdiff.append(np.max(np.abs(ue[-1]-ures[-1])))
        # plot2D(X, r.y, bShow=False)
        xMid = get_cur_mid(X, vcur) # TODO - finding middle is not trivial...
        realMid1 = X[0, np.argmax(get_exact_x(X, r.t, c, x0))] 
        realMid2 = exact_mid(r.t)
        mids.append(xMid)
        rmids1.append(realMid1)
        rmids2.append(realMid2)
        if abs(xMid - xmaxCur) > widthTol:
            # uxcur = r.y.reshape(1, M2) @ (np.linalg.lstsq(R2(X, Ps, Pbs), R1(X, Ps, Pbs))[0])
            # break; # TODO - REMOVE (just not going fowrad here)
            uxcur = np.diff(ucur)
            Xog, Xo = Xg, X
            Ps_old, Pbs_old = Ps, Pbs
            # Xg, X = adaptiveGrid.get_grid(Xo, uxcur)
            Xg, X = adaptiveGrid.get_grid((Xo[0,:-1] + Xo[0,1:])/2, uxcur)
            if np.any(Xog != Xg) or np.any(Xo != X): # if there is an actual change in the grid
                # print("DIFF:\n", Xog != Xg)
                H_and_P = get_H_and_P(Xg, X, bHO)
                Ps, Pbs = H_and_P[0], H_and_P[1]
                # plot_grid(Xg, X)
                try:
                    ucur = change_grid(J, ucur, Ps_old, Pbs_old, X, Xog, Xo, R2, bHO, bc=bc)
                except ValueError as e:
                    print(xMid)
                    print(Xo, Xog, "\n", X, Xg)
                    print(e)
                    break
                # if len(swaps) < 2:
                #     ax = plot2D(Xo, ucur, bShow=False)
                #     plot2D(X, ucur, ax=ax, bShow=False)
                #     Xop = np.hstack((0, Xo.flatten(), 1))
                #     plot2D(Xop, ue[-1], ax=ax, legend=("PREV", "NEW", "EXACT"))
                try:
                    vcur = change_grid(J, vcur, Ps_old, Pbs_old, X, Xog, Xo, R2, bHO, bc=[0, 0], bInterpolate=True) # NEED to interpolate
                except ValueError as e:
                    print(xMid)
                    print(Xo, Xog, "\n", X, Xg)
                    print(e)
                    break
                Dx = get_Dxs(R2, R1, R0, X, Ps, Pbs)
                Sc2 = S2(X, Pbs)
                Sc1 = S1(X, Pbs)
                Sc0 = S0(X, Pbs)
                # plot2D(Xo, r.y,bShow=False),plot2D(X, ucur) # TODO - remove
                r.set_initial_value(np.hstack((ucur.flatten(), vcur.flatten())), r.t)
                r.set_f_params(X, M2, Dx, Sc2, Sc1, Sc0)
                xmaxCur = xMid
                swaps.append(r.t)
    print("DONE:", toPrint)
    T = np.array(tres).reshape(len(tres), 1)
    xx, tt = np.meshgrid(np.hstack((0, X.flatten(), 1)), T)
    xx = np.array(xres)
    print ('TF=', tres[-1], "(target tf=%g)"%tf)
    U = np.array(ures)
    Ue = np.array(ue)
    print('SHAPES:', U.shape, Ue.shape)
    print('MAX diff:', np.max(mdiff))
    if bDebug:
        from matplotlib import pyplot as plt
        plt.plot(T.flatten(), mids)
        plt.plot(T.flatten(), rmids1)
        plt.plot(T.flatten(), rmids2)
        plt.plot(T.flatten(), mdiff)
        plt.plot(swaps, swaps, 'o')
        plt.legend(("CALC", "Real1", "REALREL", "ERROR", "swaps"))
        plt.ylim((0, 1))
    return xx, tt, U, Ue

# def get_cur_mid(X, ux, target=0.5):
#     # iclosest = np.argmin(np.abs(ux - target))
#     # x1 = X[0, iclosest]
#     # # print(x1, iclosest)
#     # val1 = ux[iclosest]
#     # if (val1 > target):
#     #     x2 = X[0, iclosest + 1]
#     #     val2 = ux[iclosest + 1]
#     # else:
#     #     x2 = X[0, iclosest - 1]
#     #     val2 = ux[iclosest - 1]
#     # diff1 = abs(val1 - target)
#     # diff2 = abs(val2 - target)
#     # return (diff1 * x2 + diff2 * x1)/(diff1 + diff2)
#     return X[0, np.argmin(ux)]

# def get_cur_mid(X, u):
    # Xc = (X[0,:-1] + X[0,1:])/2
    # ux = np.diff(u)
    # return Xc[np.argmin(ux)]
def get_cur_mid(X, v):
    return X[0, np.argmin(v)]


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

def change_grid(J, ucur, Pso, Pbso, X, Xog, Xo, R2, bHO, bInterpolate=True, bc=None):
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
        c1 = lambda Pbs: 2 * np.pi - Pbs[2]
        c2 = lambda Pbs: 0 * Pbs[2] + 1
        R2 = lambda X, Ps, Pbs: Ps[2] + c1(Pbs) @ X      + c2(Pbs)
        R1 = lambda X, Ps, Pbs: Ps[1] + c1(Pbs) 
        R0 = lambda X, Ps, Pbs: Ps[0]
        return R2, R1, R0
    else:
        c1 = lambda Pbs: - Pbs[2]
        c2 = lambda Pbs: 0 * Pbs[2]
        c3 = lambda Pbs: 1/6 * (- 6 * Pbs[4] + Pbs[2]) # 1/16 * (12 * np.pi) = 2 * np.pi -> S
        c4 = lambda Pbs: 0 * Pbs[2]
        R4 = lambda X, Ps, Pbs: Ps[4] + c1(Pbs) @ X**3/6  + c2(Pbs) @ X**2/2 + c3(Pbs) @ X      + c4(Pbs)
        R3 = lambda X, Ps, Pbs: Ps[3] + c1(Pbs) @ X**2/2  + c2(Pbs) @ X      + c3(Pbs)
        R2 = lambda X, Ps, Pbs: Ps[2] + c1(Pbs) @ X       + c2(Pbs)
        return R4, R3, R2


def get_Ss(X, bHO):
    if not bHO:
        c1 = lambda Pbs: 2 * np.pi
        c2 = lambda Pbs: 0
        S2 = lambda X, Pbs: c1(Pbs) * X + c2(Pbs)
        S1 = lambda X, Pbs: c1(Pbs) * X**0
        S0 = lambda X, Pbs: 0 * Pbs[2] * X**0
        return S2, S1, S0
    else:
        c1 = lambda Pbs: 0
        c2 = lambda Pbs: 0
        c3 = lambda Pbs: 2 * np.pi
        c4 = lambda Pbs: 0
        S4 = lambda X, Pbs: c1(Pbs) * X**3/6 + c2(Pbs) * X**2/2 + c3(Pbs) * X + c4(Pbs)
        S3 = lambda X, Pbs: c1(Pbs) * X**2/2 + c2(Pbs) * X      + c3(Pbs)
        S2 = lambda X, Pbs: c1(Pbs) * X      + c2(Pbs) 
        return S4, S3, S2


def get_Dxs(R2, R1, R0, X, Ps, Pbs):
    R2c = R2(X, Ps, Pbs)
    Dx = np.linalg.lstsq(R2c, R0(X, Ps, Pbs))[0]
    return Dx


def get_exact(X, T, c, x0=1/4):
    return 4 * np.arctan(np.exp((X - c * T - x0)/np.sqrt(1 - c**2)))


def get_exact_x(X, T, c, x0=1/4):
    # D(arctan(y),x) = y'/(1 + y^2)
    y = np.exp((X - c * T - x0)/np.sqrt(1 - c**2))
    yp = y/np.sqrt(1 - c**2)
    return 4 * yp/(1 + y**2)

def get_exact_t(X, T, c, x0=1/4):
    # D(arctan(y),t) = y'/(1 + y^2)
    y = np.exp((X - c * T - x0)/np.sqrt(1 - c**2))
    yp = -c * y/np.sqrt(1 - c**2)
    return 4 * yp/(1 + y**2)

def fun(t, uv, X, M2, Dx, S2, S1, S0):
    u = uv[:M2].reshape(1, M2)
    v = uv[M2:] # = dudt
    uxx = (u - S2) @ Dx + S0
    dvdt = uxx - np.sin(u)
    # if t < 4.4e-7:
    #     plot2D(X, v, ax=plot2D(X, u, bShow=False), legend=("u", "v"), title="t=%g"%t, xlims=[0, 0.4], ylims=[-.01,.01])
    return np.hstack((v, dvdt.flatten()))



if __name__ == '__main__':
    # J = 6
    alpha = 6
    beta = .4e-3
    c = .5e4
    x0 = .3
    tf = (1-2*x0)/(beta * c)
    widthTol = 1/25
    fineWidth = .2
    nrOfBorders = 1
    JRange = [4,5,6]#7]
    for J in JRange:
        mStr = "J=%d, HOHWM, fineWidth = %g"%(J, fineWidth)
        print(mStr)
        aValues = np.arange(.7, .85, .01)
        if J == 4:
            aValues = [.9,]
        elif J == 5:
            aValues = [.999,]
        elif J == 6:
            aValues = [.999,]
        for a in aValues:
            # X, T, U, Ue = solve_kdv(J, alpha=alpha, beta=beta, c=c, tf=tf, bHO=True, x0=x0, fineWidth=fineWidth, widthTol=widthTol, borders=nrOfBorders, a=a)
            X, T, U, Ue = solve_SG(J, x0=x0, a=a, widthTol=widthTol,fineWidth=fineWidth, borders=nrOfBorders)
            plot3D(X, T, U, bShow=False, title=mStr, zlims=[0, 2*np.pi]),plot3D(X,T,Ue, bShow=False),plot3D(X, T, U-Ue)