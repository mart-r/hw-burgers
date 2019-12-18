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
from utils.init_utils import get_H_and_P
from hwbasics.HwBasics import Pn_nu, Pnx_nu, Hm_nu
from utils.reprint import reprint
from utils.utils import plot2D, plot_grid, plot3D

# cluster or not
from getpass import getuser

def get_my_grid(J, lStart, rStart, borders=1):
    M2 = 2 * 2**J
    Xg = np.linspace(0, lStart, borders + 1)
    Xg = np.hstack((Xg, np.linspace(lStart, rStart, M2 - 2 * borders + 1)[1:-1]))
    Xg = np.hstack((Xg, np.linspace(rStart, 1, borders + 1)))
    X = (Xg[1:]+Xg[:-1])/2
    return Xg, X.reshape((1, M2))
    
def solve_kdv(J=3, alpha=1, beta=1, c=1000, tf=1, x0=1/4, nuStart=None, bHO=False):
    M2 = 2 * 2 ** J
    if nuStart is None:
        xStart = 1/M2
        xEnd = 1 - xStart
    else:
        xStart = nuStart
        xEnd = 1 - xStart
    Xg, X = get_my_grid(J, xStart, xEnd)
    # plot_grid(Xg, X)

    Ex = np.zeros(X.shape)
    u0 = get_exact(X, 0, alpha, beta, c, x0)
    # print('u0',u0)
    # plot2D(X, u0)

    # H, P
    H = Hm_nu(Xg)
    P2, P3 = [Pn_nu(J, nr, Xg, X.flatten()) for nr in range(2, 4)]
    P2b, P3b = [Pnx_nu(J, nr, 1, Xg) for nr in range(2, 4)]
    if bHO:
        P4, P5 = [Pn_nu(J, nr, Xg, X.flatten()) for nr in range(4, 6)]
        P4b, P5b = [Pnx_nu(J, nr, 1, Xg) for nr in range(4, 6)]

    if not bHO:
        print(P3.shape, P2b.shape, X.shape)

        c1 = 2 * P3b - 2 * P2b
        c2 = P2b - 2 * P3b
        c3 = 0 * P2b
        R3 = P3 + c1 * X**2/2 + c2 * X + c3
        R2 = P2 + c1 * X + c2
        # R3 = P3 + P2b @ (X -     X**2) + P3b @ (  X**2 - 2 * X)
        # R2 = P2 + P2b @ (1 - 2 * X)    + P2b @ (2 * Ex - 0 * Ex)
        R0 = H
        Dx1 = np.linalg.lstsq(R3, R2)[0]
        Dx3 = np.linalg.lstsq(R3, R0)[0]
    else:
        # c1 = 0 * P2b
        # c2 = 0 * P2b
        # c3 = - P3b
        # c4 = P3b - P4b
        # c5 = .5 * (-2 * P5b + 2 * P4b - P3b)    c1 = - P2_1;
        c1 = - P2b
        c2 = 0 * P2b
        c3 = 1/4 * (8 * P5b - 8 * P4b + P2b)
        c4 = 1/12 * (-24 * P5b + 12 * P4b - P2b)
        c5 = 0 * P5b
        R5 = P5 + c1 @ X**4/24 + c2 @ X**3/6 + c3 @ X**2/2 + c4 @ X + c5
        R4 = P4 + c1 @ X**3/6  + c2 @ X**2/2 + c3 @ X      + c4
        R2 = P2 + c1 @ X       + c2 @ Ex
        Dx1 = np.linalg.lstsq(R5, R4)[0]
        Dx3 = np.linalg.lstsq(R5, R2)[0]
    # in cluster or not
    cluster = getuser() == "mart.ratas"

    r = ode(fun).set_integrator('lsoda', nsteps=int(1e8))
    r.set_initial_value(u0.flatten(), 0).set_f_params(X, M2, alpha, beta, Dx1, Dx3)
    dt = tf/1e3 # TODO different? adaptive?
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
        # print('r.y', r.y)
        if np.any(r.y != r.y):
            print("'nan's in the result! CANNOT have this")
            break
    T = np.array(tres).reshape(len(tres), 1)
    xx, tt = np.meshgrid(X, T)
    Ue = get_exact(xx, tt, alpha, beta, c, x0)
    print ('TF=', tres[-1])
    U = np.array(ures)
    print('MAX diff:', np.max(np.abs(U-Ue)))
    return X.flatten(), T, U, Ue


# uinfty = 0-> c1 = c2 = c
def get_exact(X, T, alpha, beta, c, x0=1/4):
    a, b = alpha, beta
    return 3 * c * b / a * np.cosh(c**.5/2 * (X - c * b * T - x0))**(-2)


def fun(t, u, X, M2, alpha, beta, Dx1, Dx3):
    u = u.reshape(1, M2)
    ux = u @ Dx1
    uxxx = u @ Dx3
    dudt = - alpha * u * ux - beta * uxxx
    # print('t=%6.4f, umax=%10g, uxmax=%10g, uxxxmax=%10g, utmax=%10g'%(t, np.max(np.abs(u)), np.max(np.abs(ux)), np.max(np.abs(uxxx)), np.max(np.abs(dudt))))
    # if np.max(np.abs(dudt)) < 1e6:
    #     raise ValueError("NO WORKY!")
    return dudt.flatten()



if __name__ == '__main__':
    J = 7
    alpha = 6
    beta = .4e-3
    c = .5e4
    x0 = .3
    tf = (1-2*x0)/(beta * c)
    print("tf", tf)
    mStr = "J=%d,nuStart = None"%J 
    print(mStr)
    X, T, U, Ue = solve_kdv(J, alpha=alpha, beta=beta, c=c, tf=tf, bHO=False, x0=x0, nuStart=None)
    # plot3D(X, T, U, bShow=False, title=mStr),plot3D(X,T,Ue, bShow=False),plot3D(X, T, U-Ue)
    mStr = "J=%d,nuStart = .1"%J
    print(mStr)
    X, T, U, Ue = solve_kdv(J, alpha=alpha, beta=beta, c=c, tf=tf, bHO=False, x0=x0, nuStart=.1)
    # plot3D(X, T, U, bShow=False, title=mStr),plot3D(X,T,Ue, bShow=False),plot3D(X, T, U-Ue)
    mStr = "J=%d,nuStart = .2"%J
    print(mStr)
    X, T, U, Ue = solve_kdv(J, alpha=alpha, beta=beta, c=c, tf=tf, bHO=False, x0=x0, nuStart=.2)
    # plot3D(X, T, U, bShow=False, title=mStr),plot3D(X,T,Ue, bShow=False),plot3D(X, T, U-Ue)
    # HOHWM
    mStr = "J=%d,HOHWM, nuStart = None"%J
    print(mStr)
    X, T, U, Ue = solve_kdv(J, alpha=alpha, beta=beta, c=c, tf=tf, bHO=True, x0=x0, nuStart=None)
    # plot3D(X, T, U, bShow=False, title=mStr),plot3D(X,T,Ue, bShow=False),plot3D(X, T, U-Ue)
    mStr = "J=%d,HOHWM, nuStart = .1"%J
    print(mStr)
    X, T, U, Ue = solve_kdv(J, alpha=alpha, beta=beta, c=c, tf=tf, bHO=True, x0=x0, nuStart=.1)
    # plot3D(X, T, U, bShow=False, title=mStr),plot3D(X,T,Ue, bShow=False),plot3D(X, T, U-Ue)
    mStr = "J=%d,HOHWM, nuStart = .2"%J
    print(mStr)
    X, T, U, Ue = solve_kdv(J, alpha=alpha, beta=beta, c=c, tf=tf, bHO=True, x0=x0, nuStart=.2)
    # plot3D(X, T, U, bShow=False, title=mStr),plot3D(X,T,Ue, bShow=False),plot3D(X, T, U-Ue)
    # print(U-Ue)
    # plot3D(X, T, U, bShow=False),plot3D(X,T,Ue, bShow=False),plot3D(X, T, U-Ue)
    # plot2D(X, get_exact(X, T[0], 6, 1, 1e3), bShow=False),plot2D(X, get_exact(X, T[-1], 6, 1, 1e3))