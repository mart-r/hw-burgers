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
from mkdv.mkdv_adaptive_res import solve_mkdv

from datetime import datetime
import logging

import time

def get_bestest(J, scaleRange=np.arange(.3, .86, .01), wtRange=np.arange(.002, .05, .002), minValues=[0.1], x0=3./10):
        bests = []
        for scaling in scaleRange:
            for widthTol in wtRange:
                for minValue in minValues:
                    start = time.time()
                    mStr = "J=%d, HOHWM, scaling = %g, widthTol=%g, minValue=%g"%(J, scaling, widthTol, minValue)
                    print(mStr)
                    try:
                        T, U, Ue, tf = solve_mkdv(J, x0=x0, widthTol=widthTol,scaling=scaling, minWeight=minValue)[1:]
                    except ValueError as e:
                        print('Got exception (continuing on next)', e)
                        continue
                    tookTime = time.time() - start
                    md = np.max(np.abs(U - Ue))
                    print(np.max(T), md)
                    bests.append((scaling, widthTol, minValue, np.max(T), md, tookTime))
                    logging.info("%d\t%f\t%f\t%f\t%f\t%f\t%f"%(J, scaling, widthTol, minValue, np.max(T), md, tookTime))
        print ("BEST:\n", bests)
        minDiff = 1e10
        minScale = "N/A"
        minWT = "N/A"
        minWeight = "N/A"
        minTime = "N/A"
        for curScale, curWt, minValue, ctf, md, tt in bests:
            if (ctf >= tf) and (md < minDiff):
                minDiff = md
                minScale = curScale
                minWT = curWt
                minWeight = minValue
                minTime = tt
        print ("Best at scale=", minScale, "wt=", minWT, "minWeight=", minWeight, " with max diff of ", minDiff, " with calc time of ", minTime)
        return(bests)


if __name__ == "__main__":
    print('CALCULATING solve_mkdv (research)')
    now = datetime.now()
    filename = now.strftime('%Y_%m_%d_%H_%M_%S')
    if len(sys.argv) > 1:
        JRange = eval(sys.argv[1])
    else:
        JRange = [4,5,6,7]
    if len(JRange) == 1:
        filename = filename + "_J_%d"%JRange[0]
    else:
        filename = filename + "_J_%d_%d"%(JRange[0], JRange[-1])
    filename = filename + ".out"
    logging.basicConfig(filename=filename, level=logging.DEBUG)
    print("Logging to ", filename)
    for J in JRange:
        minValues = [0.2, 0.1, 0.05]
        get_bestest(J, minValues=minValues)
        # get_bestest(J, minValues=[0.1], scaleRange=[0.44, 0.4400000000000001],wtRange=[0.01])
