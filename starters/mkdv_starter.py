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
from mkdv.mkdv_adaptive_newest import solve_mkdv

from datetime import datetime
import logging


def get_bestest(J, nrOfBorders, fwRange=np.arange(.03, .5, .01), wtRange=np.arange(.001, .15, .005), aValues=[]):
        bests = []
        for fineWidth in fwRange:
            for widthTol in wtRange:
                for a in aValues:
                    mStr = "J=%d, HOHWM, fineWidth = %g, widthTol=%g, a=%g"%(J, fineWidth, widthTol, a)
                    print(mStr)
                    try:
                        T, U, Ue, tf = solve_mkdv(J, widthTol=widthTol,fineWidth=fineWidth, a=a, borders=nrOfBorders)[1:]
                    except ValueError as e:
                        print('Got exception (continuing on next)', e)
                        continue
                    md = np.max(np.abs(U - Ue))
                    print(np.max(T), md)
                    bests.append((fineWidth, widthTol, a, np.max(T), md))
                    logging.info("%d\t%f\t%f\t%f\t%f\t%f"%(J, fineWidth, widthTol, a, np.max(T), md))
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
        return(bests)


if __name__ == "__main__":
    print('CALCULATING solve_mkdv')
    now = datetime.now()
    filename = now.strftime('%Y_%m_%d_%H_%M_%S.out')
    logging.basicConfig(filename=filename, level=logging.DEBUG)
    print("Logging to ", filename)
    nrOfBorders = 1
    if len(sys.argv) > 1:
        JRange = eval(sys.argv[1])
    else:
        JRange = [4,5,6,7]
    for J in JRange:
        aValues = [1.,]
        if J == 4:
            aValues = np.arange(.881, .906, .001)
        elif J == 5:
            aValues = np.arange(.960, .980, .001)
        get_bestest(J, nrOfBorders, aValues=aValues)