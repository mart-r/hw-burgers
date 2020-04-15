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
from kdvnumiddle.calc_kdv_dynres import solve_kdv

from datetime import datetime
import logging

import time

def get_bestest(J, scaleRange=np.arange(.3, .86, .01), wtRange=np.arange(.002, .05, .002), minValues=[0.1], x0=3./10,
                alpha=6, beta=0.4e-3, c=1e3):
        bests = []
        tf = (1-2*x0)/(beta * c)
        for scaling in scaleRange:
            for widthTol in wtRange:
                for minWeight in minValues:
                    start = time.time()
                    mStr = "J=%d, HOHWM, scaling = %g, widthTol=%g, minWeight=%g"%(J, scaling, widthTol, minWeight)
                    print(mStr)
                    try:
                        T, U, Ue = solve_kdv(J, alpha=alpha, beta=beta, c=c, tf=tf, x0=x0, scaling=scaling, widthTol=widthTol, minWeight=minWeight)[1:]
                    except ValueError as e:
                        print('Got exception (continuing on next)', e)
                        continue
                    tookTime = time.time() - start
                    md = np.max(np.abs(U - Ue))
                    print(np.max(T), md)
                    bests.append((scaling, widthTol, minWeight, np.max(T), md, tookTime))
                    logging.info("%d\t%f\t%f\t%f\t%f\t%20.17f\t%f"%(J, scaling, widthTol, minWeight, np.max(T), md, tookTime))
        print ("BEST:\n", bests)
        minDiff = 1e10
        minScale = "N/A"
        minWT = "N/A"
        minWeight = "N/A"
        minTime = "N/A"
        for curScale, curWt, minWeight, ctf, md, tt in bests:
            if (ctf >= tf) and (md < minDiff):
                minDiff = md
                minScale = curScale
                minWT = curWt
                minWeight = minWeight
                minTime = tt
        print ("Best at scale=", minScale, "wt=", minWT, "minWeight=", minWeight, " with max diff of ", minDiff, " with calc time of ", minTime)
        return(bests)


if __name__ == "__main__":
    print('CALCULATING solve_mkdv (research)')
    now = datetime.now()
    filename = now.strftime('KdV_%Y_%m_%d_%H_%M_%S')
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
        minValues = [0.3, 0.25, 0.2, 0.1, 0.05]
        get_bestest(J, minValues=minValues)
        # get_bestest(J, minValues=[0.05], scaleRange=[0.43],wtRange=[0.02])
