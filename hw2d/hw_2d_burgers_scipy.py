#!/usr/bin/env python3

import numpy as np
import sys
# integration
from scipy.integrate import ode
from scipy.optimize.nonlin import newton_krylov, NoConvergence
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

# my "package"
sys.path.append('/home/mart/Documents/KybI/2019/python/NewPython2019Oct')
# my stuff
from utils.nonuniform_grid import nonuniform_grid
from hwbasics.HwBasics import Pn_nu, Pnx_nu, Hm_nu
from utils.reprint import reprint
from utils.init_utils import get_X_up_to_power, get_H_and_P

# exact
from utils.burgers_exact import exact_new_mp as exact
# # cluster or not
# from getpass import getuser

class Solver:
    # equation
    nu = None
    tf = None
    # Coordinates
    nua = None
    #x
    Jx = None
    Mx = None
    Mx2 = None
    Ex = None
    X = None
    Xg = None
    #t
    Jt = None
    Mt = None
    Mt2 = None
    Et = None
    T = None
    # integration
    #x
    Px2 = None
    Px1 = None
    Px2_1 = None
    Px1_1 = None
    Hx = None
    #t
    Pt1 = None
    Ht = None
    #combined
    Rx2 = None
    Rx1 = None
    Rx0 = None
    # IC/BC
    U0 = None
    U0x = None
    U0xx = None
    # differentiation
    Dt1 = None
    Dx1 = None
    Dx2 = None
    # last known
    lastSol = None
    lastResidual = None
    
    def __init__(self, nu, tf, Jx, nua=1, Jt=None):
        if Jt is None:
            Jt = Jx
        # equation
        self.nu = nu
        self.tf = tf
        # coordinates
        self.nua = nua
        #x
        self.Jx = Jx
        self.Mx = 2**Jx
        self.Mx2 = 2 * self.Mx
        self.Ex, self.X, self.Xg = get_X_up_to_power(Jx, nua, 1, bGet0=True)
        #y
        self.Jt = Jt
        self.Mt = 2**Jt
        self.Mt2 = 2 * self.Mt
        self.Et, self.T, self.Tg = get_X_up_to_power(Jt, nua, 1, bGet0=True) # avoid Tg (grid version)
        self.T *= self.tf
        self.Tg *= self.tf
        self.T = self.T.T
        self.Et = self.Et.T
        # integration
        #x
        self.Hx, self.Px1, self.Px2, self.Px1_1, self.Px2_1 = get_H_and_P(self.X, self.Xg, use=(1,2,'1b','2b'))
        #y
        self.Ht, self.Pt1 = get_H_and_P(self.T, self.Tg, use=(1,))
        #combined
        self.Rx2 = self.Px2 - self.Px2_1 @ self.X
        self.Rx1 = self.Px1 - self.Px2_1 @ self.Ex
        self.Rx0 = self.Hx
        # IC
        u0 = np.sin(np.pi * self.X)
        self.U0 = self.Et @ u0
        u0x = np.pi * np.cos(np.pi * self.X)
        self.U0x = self.Et @ u0x
        u0xx = - np.pi**2 * np.sin(np.pi * self.X)
        self.U0xx = self.Et @ u0xx
        # differentiation
        self.Dt1 = np.linalg.lstsq(self.Pt1, self.Ht)[0].T
        self.Dx1 = np.linalg.lstsq(self.Rx2, self.Rx1)[0]
        self.Dx2 = np.linalg.lstsq(self.Rx2, self.Rx0)[0]


    def residual(self, C):
        res = self.Ht.T @ C @ self.Rx2 + (self.Pt1.T @ C @ self.Rx2 + self.U0xx) *  (self.Pt1.T @ C @ self.Rx1) - self.nu * self.Pt1.T @ C @ self.Hx
        return res


    def residual_u(self, u):
        ux = (u - self.U0) @ self.Dx1 + self.U0x
        uxx = (u - self.U0) @ self.Dx2 + self.U0xx
        ut = self.Dt1 @ (u - self.U0)
        return ut + u * ux - self.nu * uxx

    def set_last(self, sol, res):
        self.lastSol = sol
        self.lastResidual = res

    def solve(self,):
        # guess = np.linalg.lstsq(self.Pt1.T, np.linalg.lstsq(self.Rx2.T, self.U0.T)[0].T)[0]
        # sol = newton_krylov(self.residual, guess, verbose=1)
        # print (sol)
        # print('Residual:', np.max(np.abs(self.residual(sol))))
        guess = self.U0
        try:
            sol = newton_krylov(self.residual_u, guess, verbose=1, maxiter=1000, callback=self.set_last)
        except NoConvergence:
            print ("Didn't quite work, but let's see what we've got!", np.shape(self.lastSol), np.shape(self.lastResidual))
            sol = self.lastSol.reshape(self.Mt2, self.Mx2)
        print (sol)    
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        ax = Axes3D(plt.figure(1))
        ax.plot_surface(self.X, self.T, sol)
        plt.show()
        print('Residual:', np.max(np.abs(self.residual_u(sol))))



if __name__ == '__main__':
    solver = Solver(1/(100*np.pi), .5, 5, nua=.95, Jt=3)
    # solver = Solver(1/(10), .5, 3)
    solver.solve()