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
    
def solve_mkdv(J=4, alpha=-1, beta=1e-2, mu=40, x0=1/4, fineWidth=3/16, widthTol=0.05, borders=1, a=0.9, Nper=None):
    # tf
    tf = (1 - 2 * x0) / (mu**2 * beta)
    # dt
    dt = tf/1e3 # 1000 points

    # boundary conditions
    bc = [0, 0]

    M2 = 2 * 2**J
    X, Xg = nonuniform_grid(J, 1) # TODO - use adaptive grid 
    if Nper is None:
        Nper = int(M2/4) + 1

    adaptiveGrid = AdaptiveGrid(J, AdaptiveGridType.DERIV_NU_PLUS_BORDERS, x0, borders,
                                    a=a, hw=fineWidth/2, Nper=Nper)#, bc=bc, Nper=int(M2/2)-borders)
    X, Xg = nonuniform_grid(J, 1)

    u0x = get_exact_x(X, 0, alpha, beta, mu, x0)
    # then 
    Xg, X = adaptiveGrid.get_grid(X, u0x)
    # plot_grid(Xg, X)

    Ps = get_Ps(J, X.flatten(), Xg)
    Pbs = get_Pbs(J, Xg)
    R5, R4, R3, R2 = get_R()
    S5, S4, S3, S2 = get_S()

    # x to "vector"
    X = X.reshape((1, M2))

    # matrices for differentiation
    Dx1 = np.linalg.lstsq(R5(X, Ps, Pbs), R4(X, Ps, Pbs))[0]
    Dx2 = np.linalg.lstsq(R5(X, Ps, Pbs), R3(X, Ps, Pbs))[0]
    Dx3 = np.linalg.lstsq(R5(X, Ps, Pbs), R2(X, Ps, Pbs))[0]

    # initial conditions
    u0 = get_exact(X, 0, alpha, beta, mu, x0)
    # from matplotlib import pyplot as plt
    # plt.plot(X.flatten(), u0.flatten(), '-o'),plt.show()

    # ODE solver
    solver = ode(fun).set_integrator('lsoda', nsteps=int(1e9))
    solver.set_initial_value(u0.flatten(), 0).set_f_params(alpha, beta, M2, Dx1, Dx2, Dx3, S5(X, Pbs), S4(X, Pbs), S3(X, Pbs), S2(X, Pbs))
    # start getting results
    xres = [np.hstack((0, X.flatten(), 1))]
    ures = [np.hstack((bc[0], u0.flatten(), bc[1]))]
    ueres = [np.hstack((bc[0], u0.flatten(), bc[1]))]
    tres = [0,]

    # mid
    curmid = x0

    # start integrating
    while solver.t < tf and solver.successful:
        solver.integrate(solver.t + dt)
        tres.append(solver.t)
        xres.append(np.hstack((0, X.flatten(), 1)))
        ures.append(np.hstack((bc[0], solver.y, bc[1])))
        ueres.append(np.hstack((bc[0], get_exact(X, solver.t, alpha, beta, mu, x0).flatten(), bc[1])))
        reprint('t=%g'%solver.t)
        newmid = X[0, np.argmax(solver.y)]
        if np.abs(curmid - newmid) > widthTol:
            Xog, Xo = Xg, X
            Ps_old, Pbs_old = Ps, Pbs
            uxcur = solver.y.reshape(1, M2) @ Dx1
            try:
                Xg, X = adaptiveGrid.get_grid(Xo, uxcur)
            except ValueError as e:
                print('ERROR at time', solver.t)
                raise e
            Ps, Pbs = get_Ps(J, X.flatten(), Xg), get_Pbs(J, Xg)
            ucur = change_grid(J, solver.y, Ps_old, Pbs_old, X, Xog, Xo, R3, bc=bc)
            Dx1 = np.linalg.lstsq(R5(X, Ps, Pbs), R4(X, Ps, Pbs))[0]
            Dx2 = np.linalg.lstsq(R5(X, Ps, Pbs), R3(X, Ps, Pbs))[0]
            Dx3 = np.linalg.lstsq(R5(X, Ps, Pbs), R2(X, Ps, Pbs))[0]
            solver.set_initial_value(ucur, solver.t)
            solver.set_f_params(alpha, beta, M2, Dx1, Dx2, Dx3, S5(X, Pbs), S4(X, Pbs), S3(X, Pbs), S2(X, Pbs))
            curmid = newmid
    XX = np.array(xres)
    T = np.array([tres for i in range(M2 + 2)]).T
    U = np.array(ures)
    Ue = np.array(ueres)
    return XX, T, U, Ue, tf

def change_grid(J, ucur, Pso, Pbso, X, Xog, Xo, R3, bc=[0,0], bInterpolate=True):
    if not bInterpolate:
        Px2 = np.hstack([Pnx_nu(J, 2, x, Xog) for x in X.flatten()])
        Px3 = np.hstack([Pnx_nu(J, 3, x, Xog) for x in X.flatten()])
        Px4 = np.hstack([Pnx_nu(J, 4, x, Xog) for x in X.flatten()])
        Px5 = np.hstack([Pnx_nu(J, 5, x, Xog) for x in X.flatten()])
        PsMID = [None, None, Px2, Px3, Px4, Px5]
        return ucur @ np.linalg.lstsq(R3(Xo, Pso, Pbso), R3(X, PsMID, Pbso))[0]
    else:
        filledUcur = np.hstack((bc[0], ucur.flatten(), bc[1]))
        filledX = np.hstack((0, Xo.flatten(), 1))
        interpolant = interp1d(filledX, filledUcur, 7)
        return interpolant(X.flatten()).reshape(*ucur.shape)


def fun(t, u, alpha, beta, M2, Dx1, Dx2, Dx3, S5, S4, S3, S2):
    u = u.reshape((1, M2))
    ux = (u - S5) @ Dx1 + S4
    uxxx = (u - S5) @ Dx3 + S2
    dudt = 6 * alpha * u**2 * ux - beta * uxxx
    return dudt.flatten()


def get_Ps(J, X, Xg):
    P5 = Pn_nu(J, 5, Xg, X)
    P4 = Pn_nu(J, 4, Xg, X)
    P3 = Pn_nu(J, 3, Xg, X)
    P2 = Pn_nu(J, 2, Xg, X)
    P1 = Pn_nu(J, 1, Xg, X)
    P0 = Hm_nu(Xg)
    return P0, P1, P2, P3, P4, P5

def get_Pbs(J, Xg, xAt=1):
    Pb5 = Pnx_nu(J, 5, xAt, Xg)
    Pb4 = Pnx_nu(J, 4, xAt, Xg)
    Pb3 = Pnx_nu(J, 3, xAt, Xg)
    Pb2 = Pnx_nu(J, 2, xAt, Xg)
    Pb1 = Pnx_nu(J, 1, xAt, Xg)
    return None, Pb1, Pb2, Pb3, Pb4, Pb5

def get_R():
    bc3_is_left = False
    if bc3_is_left: # if the third boundary condition is u_x(0, t) = 0 
        c1 = lambda Pbs: - Pbs[2]
        c2 = lambda Pbs: 0 * Pbs[4]
        c3 = lambda Pbs: - 2 * Pbs[5] + 1/12. * Pbs[2]
        c4 = lambda Pbs: 0 * Pbs[4]
        c5 = lambda Pbs: 0 * Pbs[4]
        R5 = lambda X, Ps, Pbs: Ps[5] + c1(Pbs) @ X**4/24 + c2(Pbs) @ X**3/6 + c3(Pbs) @ X**2/2 + c4(Pbs) @ X + c5(Pbs)
        R4 = lambda X, Ps, Pbs: Ps[4] + c1(Pbs) @ X**3/6  + c2(Pbs) @ X**2/2 + c3(Pbs) @ X      + c4(Pbs)
        R3 = lambda X, Ps, Pbs: Ps[3] + c1(Pbs) @ X**2/2  + c2(Pbs) @ X      + c3(Pbs) 
        R2 = lambda X, Ps, Pbs: Ps[2] + c1(Pbs) @ X       + c2(Pbs) 
    else: # third boundary conditions is u_x(1,t) = 0 
        c1 = lambda Pbs: - Pbs[2]
        c2 = lambda Pbs: 0 * Pbs[4]
        c3 = lambda Pbs: 2 * Pbs[5] - 2 * Pbs[4] + 1/4. * Pbs[2]
        c4 = lambda Pbs: -2 * Pbs[5] + Pbs[4] - 1/12. * Pbs[2]
        c5 = lambda Pbs: 0 * Pbs[4]
        R5 = lambda X, Ps, Pbs: Ps[5] + c1(Pbs) @ X**4/24 + c2(Pbs) @ X**3/6 + c3(Pbs) @ X**2/2 + c4(Pbs) @ X + c5(Pbs)
        R4 = lambda X, Ps, Pbs: Ps[4] + c1(Pbs) @ X**3/6  + c2(Pbs) @ X**2/2 + c3(Pbs) @ X      + c4(Pbs)
        R3 = lambda X, Ps, Pbs: Ps[3] + c1(Pbs) @ X**2/2  + c2(Pbs) @ X      + c3(Pbs) 
        R2 = lambda X, Ps, Pbs: Ps[2] + c1(Pbs) @ X       + c2(Pbs) 
    return R5, R4, R3, R2

def get_S():
    c1 = lambda Pbs: 0
    c2 = lambda Pbs: 0
    c3 = lambda Pbs: 0
    c4 = lambda Pbs: 0
    c5 = lambda Pbs: 0
    S5 = lambda X, Pbs: c1(Pbs) * X**4/24 + c2(Pbs) * X**3/6 + c3(Pbs) * X**2/2 + c4(Pbs) * X + c5(Pbs)
    S4 = lambda X, Pbs: c1(Pbs) * X**3/6  + c2(Pbs) * X**2/2 + c3(Pbs) * X      + c4(Pbs)
    S3 = lambda X, Pbs: c1(Pbs) * X**2/2  + c2(Pbs) * X      + c3(Pbs) 
    S2 = lambda X, Pbs: c1(Pbs) * X       + c2(Pbs) 
    return S5, S4, S3, S2

def get_exact(X, T, alpha, beta, mu, x0=1/4):
    h1 = np.sqrt(-beta/alpha) * mu
    H1 = mu * ( X - T * mu**2 * beta - x0)
    return h1 * np.cosh(H1)**(-1)

def get_exact_x(X, T, alpha, beta, mu, x0=1/4):
    h1 = np.sqrt(-beta/alpha) * mu**2
    H1 = mu * ( X - T * mu**2 * beta - x0)
    return h1 * np.cosh(-H1)**(-1) * np.tanh(-H1)

if __name__ == "__main__":
    X, T, U, Ue, tf = solve_mkdv(J=7, x0=3/10., fineWidth=0.25, widthTol=0.001, a=1.0, mu=120)
    print('TF=', np.max(T), 'max diff:', np.max(np.abs(U - Ue)))
    print(X.shape, T.shape, U.shape, Ue.shape)
    plot3D(X, T, U, bShow=False)
    plot3D(X, T, Ue, bShow=False)
    plot3D(X, T, U - Ue)