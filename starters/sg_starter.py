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
from sinegordon.sgadaptive import solve_SG


if __name__ == "__main__":
    x0 = .3
    c = 1 - 1e-4
    tf = (1 - 2 * x0)/c
    nrOfBorders = 1
    JRange = [4,5,6,7]
    for J in JRange:
        bests = []
        for fineWidth in np.arange(.2, .5, .01):
            for widthTol in np.arange(.04, .15, .005):
                if J == 4:
                    aValues = np.arange(.881, .906, .001)
                elif J == 5:
                    aValues = np.arange(.960, .980, .001)
                elif J == 6:
                    aValues = np.arange(1, 1.0001, .001)
                elif J == 7:
                    aValues = [1.,]
                for a in aValues:
                    mStr = "J=%d, HOHWM, fineWidth = %g, widthTol=%g, a=%g"%(J, fineWidth, widthTol, a)
                    print(mStr)
                    try:
                        X, T, U, Ue = solve_SG(J, c=c, x0=x0, a=a, widthTol=widthTol,fineWidth=fineWidth, borders=nrOfBorders)
                    except ValueError as e:
                        print('Got exception (continuing on next)', e)
                    md = np.max(np.abs(U - Ue))
                    bests.append((fineWidth, widthTol, a, np.max(T), md))
        print ("BEST:\n", bests)
        minDiff = 1e10
        minFW = "N/A"
        minWT = "N/A"
        minA = "N/A"
        for fw, wt, a, ctf, md in bests:
            if (ctf >= tf) and (md < minDiff):
                minDiff = md
                minFW = fw
                minWT = wt
                minA = a
        print ("Best at fw=", minFW, "wt=", minWT, "a=", minA, " with max diff of ", minDiff)