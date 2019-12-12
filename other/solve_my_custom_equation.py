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
from utils.init_utils import get_H_and_P, get_X_up_to_power
from hwbasics.HwBasics import Pn_nu, Pnx_nu, Hm_nu
from utils.reprint import reprint

# cluster or not
from getpass import getuser


def my_solve(J, tf, d1=20, d2=1/10., bHO=False, hos=1, nua=1):
    """Trying to solve my own custom differential (started 21.11.19)
    (this part 12.12.19)
    The equation: utt = x^2/(d2+t)^2 uxx + ux + 2 d1 (d2 + t) tanh(d1 (d2 + t) x) u
    To solve, using system:
         - ut = v
         - vt = x^2/(d2+t)^2 uxx + ux + 2 d1 (d2 + t) tanh(d1 (d2 + t) x) u
    Using boundary conditions: u(0,t) = 1 and u(1,t) = 0
    d1 - the 'steepness' parameter
    d2 - initial excitation parameter (0 -> IC=0, the higher the faster the steep slope is obtained)
    General solution:
    u = sech(d1 ( d2 + t) x)^2
    """
    M = 2**J
    M2 = 2 * M

    Ex, X, Xg = get_X_up_to_power(J, nua, 1, True)

    if not bHO:
        H, P1, P2, P1b, P2b = get_H_and_P(X, Xg, use=(1, 2, '1b', '2b'))
    else:
        if hos == 1:
            H, P1, P2, P3, P4, P2b, P3b, P4b = get_H_and_P(X, Xg, use=(1, 2, 3, 4, '2b', '3b', '4b'))
        elif hos >= 2:
            raise Exception("Not implemented!")

    u0 = get_exact(d1, d2, X, 0) # INITIAL CONDITION
    v0 = get_exact_t(d1, d2, X, 0)  # INITIAL VELOCITY


    # Dx1:u->ux, Dx2:u->uxx
    if bHO:
        c1 = lambda t: 1./(-4 + d2**2 + 2 * d2 * t + t**2) * (
                    P4b * (6 * (-2 + 2 * d2**2 + 4 * d2 * t + 2 * t**2)) +
                    P3b * (6 * (-d2**2 - 2 * d2 * t - t**2)) +
                    P2b *  6
        )
        c2 = lambda t: 1./(-4 + d2**2 + 2 * d2 * t + t**2) * (
                    P4b * (-2 * (-6 + 3 * d2**2 + 6 * d2 * t + 3 * t**2)) +
                    P3b * (-2 * (-d2**2 - 2 * d2 * t - t**2)) +
                    P2b * (-2)
        )
        c3 = lambda t: P2b *  0
        c4 = lambda t: P2b *  0
        if hos == 1:
            # print(P4.shape, c1(0).shape, X.shape, c2(0).shape, X.shape, c3(0).shape, X.shape, c4(0).shape, Ex.shape)
            # print(P4 + 
            #         c1(1e-9) @ X**3/6 + 
            #         c2(1e-9) @ X**2/2 + 
            #         c3(1e-9) @ X + 
            #         c4(1e-9) @ Ex)
            R4 = lambda t: P4 + c1(t) @ X**3/6 + c2(t) @ X**2/2 + c3(t) @ X + c4(t) @ Ex
            R3 = lambda t: P3 + c1(t) @ X**2/2 + c2(t) @ X      + c3(t) @ Ex
            R2 = lambda t: P2 + c1(t) @ X      + c2(t) @ Ex
        else:
            raise Exception("Not implemented!")
        Dx1 = lambda t: np.linalg.lstsq(R4(t), R3(t))[0]
        Dx2 = lambda t: np.linalg.lstsq(R4(t), R2(t))[0]
        # print('Dx1', Dx1(0), 'Dx1(1e-3)', Dx2(1e-3))
        # S 
        cc1 = lambda t: 6 * (-2 + 2 * d2**2 + 4 * d2 * t + 2 * t**2)/(-4 + d2**2 + 2 * d2 * t + t**2)
        cc2 = lambda t:-2 * (-6 + 3 * d2**2 + 6 * d2 * t + 3 * t**2)/(-4 + d2**2 + 2 * d2 * t + t**2)
        cc3 = lambda t: 0
        cc4 = lambda t: 1
        S2 = lambda t: cc1(t) * X**3/6 + cc2(t) * X**2/2 + cc3(t) * X + cc4(t) * Ex
        S1 = lambda t: cc1(t) * X**2/2 + cc2(t) * X      + cc3(t) * Ex
        S0 = lambda t: cc1(t) * X      + cc2(t) * Ex
    else:
        c1 = - P2b
        c2 = 0 * P1b
        R2 = P2 + c1 @ X + c2 @ Ex
        R1 = P1 + c1 @ Ex
        R0 = H
        Dx1 = lambda t: np.linalg.lstsq(R2, R1)[0]
        Dx2 = lambda t: np.linalg.lstsq(R2, R0)[0]
        # S
        c1 = lambda t: -1
        c2 = lambda t: 1
        S2 = lambda t: c1(t) * X + c2(t) * Ex
        S1 = lambda t: c1(t) * Ex
        S0 = lambda t: 0     * Ex

    # in cluster or not
    cluster = getuser() == "mart.ratas"

    r = ode(fun).set_integrator('lsoda', nsteps=int(1e8))
    r.set_initial_value(np.hstack([u0.flatten(),v0.flatten()]), 0).set_f_params(X, M2, d1, d2, Dx1, Dx2, S2, S1, S0)
    dt = 1e-3 # TODO different? adaptive?
    tres = [0]
    ures = [u0.flatten()]
    # print("R4(0)=", R4(0))
    # print('Dx1(0)=', Dx1(0))
    # print('BEFORE, fun:', fun(0, np.hstack([u0.flatten(),v0.flatten()]), X, M2, d1, d2, Dx1, Dx2, S2, S1, S0))
    while r.t < tf:
        r.integrate(r.t+dt)
        toPrint = "%g" % r.t
        if cluster:
            print(toPrint)
        else:
            reprint(toPrint)
        tres.append(r.t)
        ures.append(r.y[:M2]) # reshape?
        if (not r.successful and np.max(r.y) > 1e5) or len(tres) > tf/dt: # TODO - this is a "weird" limit
            break
        # print('r.y', r.y)
        if np.any(r.y != r.y):
            print("'nan's in the result! CANNOT have this")
            break
    T = np.array(tres).reshape(len(tres), 1)
    Et = np.ones(T.shape)
    xx = Et @ X
    tt = T @ Ex
    Ue = get_exact(d1, d2, xx, tt)
    print ('TF=', tres[-1])
    return X.flatten(), T, ures, Ue

counter = 0
def fun(t, uv, X, M2, d1, d2, Dx1, Dx2, S2, S1, S0):
    """
    SYSTEM:
         - ut = v
         - vt = x^2/(d2+t)^2 uxx + ux + 2 d1 (d2 + t) tanh(d1 (d2 + t) x) u
    """
    # print('in fun',t)#, u)
    u = uv[:M2].reshape(1, M2)
    v = uv[M2:]
    # print(t, u.shape, S2(t).shape, Dx1(t).shape,S1(t).shape)
    ux = (u - S2(t)) @ Dx1(t) + S1(t)
    uxx = (u - S2(t)) @ Dx2(t) + S0(t)
    # print ('ux', ux)
    if np.any(ux == 0):
        print('0 in ux:', ux)
    # global counter
    # counter += 1 
    # step = 5000
    # if counter % step == 0 and counter < 10 * step:
    #     from matplotlib import pyplot as plt
    #     plt.plot(X.flatten(), u.flatten())
    #     plt.title('i=%d, t=%g'%(counter,t))
    #     plt.show()
    
    dvdt = X**2/(d2 + t)**2 * uxx + ux + 2 * d1 * (d2 + t) * np.tanh(d1 * (d2 + t) * X) * u
    if np.any(dvdt != dvdt):
        print('t=', t)
        print('nan', dvdt)
        print('u',u)
        print('ux', ux)
        print('uxx', uxx)
        # from matplotlib import pyplot as plt
        # plt.plot(u.flatten())
        # plt.show()
        # raise ValueError("NANS")
        dvdt = np.zeros(dvdt.shape)
    return np.hstack([v,dvdt.flatten()])

def find_error_in_time(X, T, U, Ue, bShow=True, tol=1e-3):
    in_time = []
    last_index = 0
    alreadyOver = False
    for u, ex in zip(U, Ue):
        cmdiff = np.max(np.abs(u-ex))
        in_time.append(cmdiff)

        if not alreadyOver and cmdiff < tol:
            last_index += 1
        if cmdiff >= tol:
            alreadyOver = True
    # in_time = np.array(in_time)
    # for last_index, u, ex in zip(range(len(T)), U, Ue):
    #     if np.max(np.abs())
    print ('ERROR at t=%5.3f is %g'%(T[last_index], in_time[last_index]))
    if bShow:
        from matplotlib import pyplot as plt
        plt.plot(T.flatten(), in_time)
        plt.title('Max. deviation from exact in time')
        plt.plot([0,1],[tol,tol], color='red')
        plt.ylim([0,2 * tol])
        plt.show()
    return in_time

def show_results(X, T, U, Ue):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    X, T = np.meshgrid(X, T)
    U = np.array(U)
    print(type(X),type(T),type(U),type(Ue))
    print(X.shape, T.shape, U.shape, Ue.shape)
    #calc
    ax = Axes3D(plt.figure())
    ax.plot_surface(X, T, U)
    ax.set_title('CALC')
    #exact
    ax = Axes3D(plt.figure())
    ax.plot_surface(X, T, Ue)
    ax.set_title('EXACT')
    #diff
    ax = Axes3D(plt.figure())
    ax.plot_surface(X, T, U-Ue)
    ax.set_title('DIFF')

    plt.show()


def plot_results(X, T, U, Ue, bShow=True, nrSkip=None):
    if nrSkip is None:
        nrSkip = int(len(U)/10)
    print('Showing results!')
    from matplotlib import pyplot as plt
    i = 0
    mdiff = 0
    for t, u, e in zip(T, U, Ue): # need to reshape e
        if i%nrSkip == 0 and bShow:
            plt.figure()
            plt.plot(X, u.flatten())
            plt.title('t=%f'%t)
            print('title:', 't=%f'%t)
        i += 1
        mdiff = max(mdiff, np.max(u-e))
    if bShow:
        plt.show()
    print('Showed results!, maxDIFF:', mdiff)

# EXACT:
def get_exact(d1, d2, X, T):
    return np.cosh(d1 * (d2 + T) * X)**(-2)

# EXACT_t (initial velocity):
def get_exact_t(d1, d2, X, T):
    return - 2 * d1 * X * np.cosh(d1 * (d2 + T) * X)**(-2) * np.tanh(d1 * (d2 + T) * X)    


def saver(fileName, X, T, U, Ue):
    from utils.utils import save
    save(fileName, X=X, T=T, U=U, Ue=Ue)


if __name__ == '__main__':
    print('starting(AlmostSingular)')
    tf = 1
    for nua in [1.0, 1.05, 1.1, 1.15, 1.2, 1.25,1.3, 1.35]:
        print('nua=', nua)
        [X, T, U, Ue] = my_solve(4, tf, 60, 1/10., bHO=True, hos=1, nua=nua)
        # plot_results(X, T, U, Ue, bShow=True)
        find_error_in_time(X, T, U, Ue)
        # show_results(X, T, U, Ue)
        fileName = "HW_almost_singular_test"
        saver(fileName, X, T, U, Ue)