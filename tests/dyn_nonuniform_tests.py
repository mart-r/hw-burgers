#!/usr/bin/env python3

import sys, getopt
print (sys.path[0])
sys.path.append('/home/mart/Documents/KybI/2019/python/NewPython2019Oct')
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")
from dynnu.dynamic_nonuniform_grid import DynGridHandler
import numpy as np

def test_differentiation():
    handler = DynGridHandler(5, 3, 1)
    for i in range(handler.max_i):
        print(i)
        Xstuff, H_and_P = handler.get_for_grid(i)
        X, Xg = Xstuff
        H, P1, P2, P1b, P2b = H_and_P
        R2 = P2 - P2b @ X
        R1 = P1 - P2b @ (X * 0 + 1)
        R0 = H
        Dt1 = np.linalg.lstsq(R2, R1)[0]
        Dt2 = np.linalg.lstsq(R2, R0)[0]
        # u
        u = np.sin(2*np.pi*X)
        uxe = 2 * np.pi * np.cos(2*np.pi*X)
        uxc = u @ Dt1
        # diff _x
        mdx1 = np.max(np.abs(uxe-uxc))
        print('Dx', mdx1)
        uxxe = -4 * np.pi**2 * np.sin(2*np.pi*X)
        uxxc = u @ Dt2
        # diff _xx
        mdx2 = np.max(np.abs(uxxe-uxxc))
        print('Dxx', mdx2)
        # if (mdx1 > 1 or mdx2 > 1):
        #     from matplotlib import pyplot as plt
        #     plt.plot(X.flatten(), u.flatten())
        #     plt.figure()
        #     plt.plot(X.flatten(), uxe.flatten())
        #     plt.plot(X.flatten(), uxc.flatten())
        #     plt.figure()
        #     plt.plot(X.flatten(), uxxe.flatten())
        #     plt.plot(X.flatten(), uxxc.flatten())
        #     plt.show()


def test_integration():
    handler = DynGridHandler(5, 3, 1)
    for i in range(handler.max_i):
        print(i)
        Xstuff, H_and_P = handler.get_for_grid(i)
        X, Xg = Xstuff
        H, P1, P2, P1b, P2b = H_and_P
        R2 = P2 - P2b @ X
        R1 = P1 - P2b @ (X * 0 + 1)
        R0 = H
        Ix1 = np.linalg.lstsq(H, R1)
        Ix2 = np.linalg.lstsq(H, R2)
        # u
        uxx = np.sin(2*np.pi*X)
        # int x 1
        uxe = - 1/(2*np.pi) * np.cos(2*np.pi*X)
        uxc = uxx * Ix1
        # diff
        mdix1 = np.max(np.abs(uxe-uxc))
        print('Ix', mdix1)
        # int x 2
        ue = - 1/(4*np.pi**2) * np.sin(2*np.pi*X)
        uc = uxx * Ix2
        # diff
        mdix2 = np.max(np.abs(ue-uc))
        print('Ixx', mdix2)
        if (mdix1 > 1 or mdix2 > 1):
            from matplotlib import pyplot as plt
            plt.plot(X.flatten(), u.flatten())
            plt.figure()
            plt.plot(X.flatten(), uxe.flatten())
            plt.plot(X.flatten(), uxc.flatten())
            plt.figure()
            plt.plot(X.flatten(), uxxe.flatten())
            plt.plot(X.flatten(), uxxc.flatten())
            plt.show()


if __name__ == '__main__':
    test_differentiation()