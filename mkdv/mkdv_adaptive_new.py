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
    
def solve_mkdv(J=4, la=50, delta=1./50, x0=3/4, fineWidth=3/16, widthTol=0.05, borders=1, a=0.9):
    # tf
    # tf = (4 * x0 - 2)/(la**2 * delta)
    tf = 1e-5
    # dt
    dt = tf/1e3 # 1000 points

    M2 = 2 * 2**J
    X, Xg = nonuniform_grid(J, 1) # TODO - use adaptive grid 
    Ps = get_Ps(J, X, Xg)
    Pbs = get_Pbs(J, Xg)
    R5, R4, R3, R2 = get_R(la)
    S5, S4, S3, S2 = get_S(la, delta)

    # x to "vector"
    X = X.reshape((1, M2))

    # matrices for differentiation
    Dx1 = np.linalg.lstsq(R5(X, Ps, Pbs), R4(X, Ps, Pbs))[0]
    Dx2 = np.linalg.lstsq(R5(X, Ps, Pbs), R3(X, Ps, Pbs))[0]
    Dx3 = np.linalg.lstsq(R5(X, Ps, Pbs), R2(X, Ps, Pbs))[0]

    # initial conditions
    u0 = get_exact(X, 0, la, delta, x0=x0)
    # boundary conditions
    bc = [-get_bc(la, delta), get_bc(la, delta)]

    # ODE solver
    solver = ode(fun).set_integrator('lsoda', nsteps=int(1e9))
    solver.set_initial_value(u0.flatten(), 0).set_f_params(delta, M2, Dx1, Dx2, Dx3, S5(X, Pbs), S4(X, Pbs), S3(X, Pbs), S2(X, Pbs))
    # start getting results
    xres = [np.hstack((0, X.flatten(), 1))]
    ures = [np.hstack((bc[0], u0.flatten(), bc[1]))]
    ueres = [np.hstack((bc[0], u0.flatten(), bc[1]))]
    tres = [0,]

    # start integrating
    while solver.t < tf and solver.successful:
        solver.integrate(solver.t + dt)
        tres.append(solver.t)
        xres.append(np.hstack((0, X.flatten(), 1)))
        ures.append(np.hstack((bc[0], solver.y, bc[1])))
        ueres.append(np.hstack((bc[0], get_exact(X, solver.t, la, delta, x0=x0).flatten(), bc[1])))
        reprint('t=%g'%solver.t)
    XX = np.array(xres)
    T = np.array([tres for i in range(M2 + 2)]).T
    U = np.array(ures)
    Ue = np.array(ueres)
    return XX, T, U, Ue


def fun(t, u, delta, M2, Dx1, Dx2, Dx3, S5, S4, S3, S2):
    u = u.reshape((1, M2))
    ux = (u - S5) @ Dx1 + S4
    uxxx = (u - S5) @ Dx3 + S2
    dudt = u**2 * ux - delta * uxxx
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

def get_bc(la, delta):
    return np.sqrt(3./2 * delta) * la

def get_R(la):
    c1 = lambda Pbs: (24 * la**2 * Pbs[5] - 12 * la**2 * Pbs[4] + 8 * Pbs[2])/(-8 + la**2)
    c2 = lambda Pbs: 0 * Pbs[4]
    c3 = lambda Pbs: (- (-48 + 12 * la**2)/3. * Pbs[5] + la**2 * Pbs[4] - 2/3 * Pbs[2])/(-8 + la**2)
    c4 = lambda Pbs: 0 * Pbs[4]
    c5 = lambda Pbs: 0 * Pbs[2]
    R5 = lambda X, Ps, Pbs: Ps[5] + c1(Pbs) @ X**4/24 + c2(Pbs) @ X**3/6 + c3(Pbs) @ X**2/2 + c4(Pbs) @ X + c5(Pbs)
    R4 = lambda X, Ps, Pbs: Ps[4] + c1(Pbs) @ X**3/6  + c2(Pbs) @ X**2/2 + c3(Pbs) @ X      + c4(Pbs)
    R3 = lambda X, Ps, Pbs: Ps[3] + c1(Pbs) @ X**2/2  + c2(Pbs) @ X      + c3(Pbs) 
    R2 = lambda X, Ps, Pbs: Ps[2] + c1(Pbs) @ X       + c2(Pbs) 
    return R5, R4, R3, R2

def get_S(la, delta):
    c1 = lambda Pbs: (-24 * np.sqrt(6 * delta) * la**3)/(-8 + la**2)
    c2 = lambda Pbs: 0
    c3 = lambda Pbs: -(48 * np.sqrt(6 * delta) * la - 12 * np.sqrt(6 * delta) * la**3)/(3 * (-8 + la**2))
    c4 = lambda Pbs: 0
    c5 = lambda Pbs: - np.sqrt(3./2 * delta) * la
    S5 = lambda X, Pbs: c1(Pbs) * X**4/24 + c2(Pbs) * X**3/6 + c3(Pbs) * X**2/2 + c4(Pbs) * X + c5(Pbs)
    S4 = lambda X, Pbs: c1(Pbs) * X**3/6  + c2(Pbs) * X**2/2 + c3(Pbs) * X      + c4(Pbs)
    S3 = lambda X, Pbs: c1(Pbs) * X**2/2  + c2(Pbs) * X      + c3(Pbs) 
    S2 = lambda X, Pbs: c1(Pbs) * X       + c2(Pbs) 
    return S5, S4, S3, S2

def get_exact(X, T, l, d, x0=3/4):
    h1 = l/2 * np.sqrt(6 * d)
    H1 = l/2 * (X + d/2 * l**2 * T - x0)
    return h1 * np.tanh(H1)

def get_exact_x(X, T, l, d, x0=1/4):
    h1 = 1/2. * np.sqrt(3/2.)  * np.sqrt(d) * l**2
    H1 = l/2 * (X + d/2 * l**2 * T - x0)
    return h1 * np.cosh(H1)**(-2)

if __name__ == "__main__":
    X, T, U, Ue = solve_mkdv(J=5, x0=1/4.)
    print('TF=', np.max(T), 'max diff:', np.max(np.abs(U - Ue)))
    print(X.shape, T.shape, U.shape, Ue.shape)
    plot3D(X, T, U, bShow=False)
    plot3D(X, T, Ue, bShow=False)
    plot3D(X, T, U - Ue)