#!/usr/bin/env python3

from numpy import pi # in case you want pi (i.e in nu evaluation)
import sys
# my "package"
sys.path.append('/home/mart/Documents/KybI/2019/python/NewPython2019Oct')
from hw2d.hw_2d_burgers import hw_2d_burgers as solver
from higherlevel.hw_euler_burgers import saver, plot_results, get_exact
from starters.hw_eluer_burgers_starter import parse_arguments
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")


if __name__ == '__main__':
    args = sys.argv[1:]
    u01 = 1 # not changing, for now at least
    L = 1   # not changing, for now at least
    JAll, nuAll, tf, bHO, nua, bFindExact = parse_arguments(args)
    for J in JAll:
        for nu in nuAll:
            print('Jx, nu, tf, bHO, nua, bFindExact')
            print(J, nu, tf, bHO, nua, bFindExact)
# hw_2d_burgers(Jx, nu, nua=1, bHO=False, bfindExact=True, tf=1/2, u0i=1, L=1, Jy=None, rMax=15, tol=1e-10):
            X, T, U = solver(J, nu=nu, tf=tf, u0i=u01, L=L, bHO=bHO, nua=nua, Jy=3)
            if bFindExact:
                Ue = get_exact(nu, X, T.flatten(), bHighDPS=abs(nu)<1/50).T
                # find suitable nu for first timestep
                u1 = U[0,:]
                t = T[0]
                cmaxdiff = 1e3
                cnu = nu
                # print('FINDING SUITABLE nu')
                # while cmaxdiff > 0.5e-3:
                #     import numpy as np
                #     u1e = get_exact(cnu, X, T.flatten(), bHighDPS=abs(cnu)<1/50).T
                #     cmaxdiff = np.max(np.abs(u1e-u1))
                #     cnu -= 1e-5
                #     print(cnu,cmaxdiff)
            fname = 'n_HW2D_Burgers_AS_J=%d_nu=%f_tf=%d_nua=%f'%(J, nu, tf, nua)
            if bHO:
                fname += "_HO"
            fname += ".mat"
            plot_results(X, T.flatten(), U ,Ue, False)
            saver(fname, X, T, U, Ue)
