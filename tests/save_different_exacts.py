#!/usr/bin/env python3

import numpy as np
import sys, getopt
print (sys.path[0])
sys.path.append('/home/mart/Documents/KybI/2019/python/hw-burgers')
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

from utils.burgers_exact import exact_new_mp as exact
from utils.utils import save
from utils.hideprint import HiddenPrints as HP

def calc_exact(X, t, nu):
    bHighDPS = nu < (1/100)
    infty = 200
    if bHighDPS:
        import mpmath
        mpmath.mp.dps = 800
        infty = 800
    else:
        import mpmath
        mpmath.mp.dps = 200
    return exact(X, t, nu, infty=infty)

def calc_exact_around(X, t, calc, nu, before=1e-3, after=1e-4, step=1e-5):
    nuStart = nu - before
    nuStop = nu + after
    nuOrig = nu
    others = {}
    print(nuStart, nu, nuStop)
    tv = np.array([t, t])
    for nu in np.arange(nuStart, nuStop + step, step):
        print('doing for nu=',nu, end='')
        with HP():
            cur = calc_exact(X, tv, nu)[:, 0]
            others[nu] = cur
            maxDiff = np.max(np.abs(cur - calc))
        print(', max diff:', maxDiff)
    # save('exacts_for_nu_at_time.mat', X=X, t=t, nu=nuOrig, ) # todo - float values - how?



if __name__ == '__main__':
    filename = "/home/mart/Documents/KybI/2019/python/NewPython2019Oct/hw2d_burgers_newton_krylov_Jx=3_Jt=4_nu=0.003183_tf=0.500000_nuax=0.750000_nuat=0.900000.mat"
    from scipy.io import loadmat
    inD = loadmat(filename)
    X = inD['X'][0]
    t = inD['T'][0][0]
    calc = inD['U'][0, :]
    nu = inD['nu'][0,0]
    print(X,t,calc,nu)
    calc_exact_around(X, t, calc, nu, -0.00028, 2e-3)

