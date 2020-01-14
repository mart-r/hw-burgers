#!/usr/bin/env python3

import sys, getopt
print (sys.path[0])
sys.path.append('/home/mart/Documents/KybI/2019/python/hw-burgers')
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")
import numpy as np
from kdvnumiddle.calc_kdv_dyn import get_smart_grid_left_right

def do_test(N=4, skip=0.2, step=0.05):
    print("N=%d"%N)
    for left in np.arange(0, 1 - skip + step, step):
        right = 1 - left - skip
        nrl, nrr = get_smart_grid_left_right(N, left, right)
        print ("%2d, %2d"%(nrl, nrr), nrl+nrr)


if __name__ == '__main__':
    do_test(2)
    do_test(4)
    do_test(6)
    do_test(8)
    do_test(10)