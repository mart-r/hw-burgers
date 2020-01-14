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
from hwbasics.HwBasics import Pn_nu, Pnx_nu, Hm_nu
from utils.reprint import reprint
from utils.init_utils import get_X_up_to_power, get_H_and_P

class DynGridHandler:
    def __init__(self, Jf, Jc, max_x_power=3): # n_max = max_x_power + 1 -> Px_{n_max}
        # fine grid
        self.Jf = Jf
        self.Mf = 2**Jf
        self.Mf2 = 2 * self.Mf
        Xf_and_power = get_X_up_to_power(self.Jf, 1, max_x_power, True) # UNIFORM fine grid
        self.Ex = Xf_and_power[0]
        self.X = Xf_and_power[1]
        self.Xg = Xf_and_power[-1]
        self.X_powers = Xf_and_power[:-1] # inclduing 0th and 1st power (duplicates)
        use = [el for el in range(1, max_x_power + 2)] + [str(el) + "b" for el in range(1, max_x_power + 2)]
        # coarse grid
        self.Jc = Jc
        self.Mc = 2**Jc
        self.Mc2 = 2 * self.Mc
        # find all coarse grids
        self.delta = int(2 * self.Mf/self.Mc - 1)
        self.i0 = int(self.delta/2)
        self.max_i = int(self.Mf2 - self.Mc)
        self.grids = [self.get_grid(i) for i in range(self.max_i)]
        self.H_and_P = []
        for X, Xg in self.grids:
            use = [el for el in range(1,max_x_power +2)] + [str(el) + 'b' for el in range(1,max_x_power +2)]
            self.H_and_P.append(get_H_and_P(X.flatten(), Xg, use=use))
        print('GRIDS:', len(self.grids), len(self.grids[0]), len(self.grids[0][0]))
        print("H_and_P:", len(self.H_and_P), "H_and_P[0]:", len(self.H_and_P[0]), "H_and_P[0][0]", len(self.H_and_P[0][0])) # DEBUG

    def get_grid(self, i):
        Xf = self.X.flatten()
        X = []
        icur = self.i0
        while icur < i:
            X.append(Xf[icur])
            icur += self.delta
        X += list(Xf[i:i+self.Mc])
        icur = i + self.Mc - 1 + self.i0
        while icur < self.Mf2 and len(X) < self.Mc2:
            X.append(Xf[icur])
            icur += self.delta
        Xg = [0,]
        for x in X:
            Xg.append(x + (x - Xg[-1]))
        return np.array(X).reshape(1, self.Mc2), np.array(Xg)

    def get_for_grid(self, i):
        return self.grids[i], self.H_and_P[i]

    def get_fine_grid(self):
        return self.Xg, self.X


if __name__ == '__main__':
    Jf = 5
    Mf = 2**Jf
    Jc = 3
    Mc = 2**Jc
    handler = DynGridHandler(Jf, Jc)
    
    XoPlus, H_and_P = handler.get_for_grid(int())
    Xo, Xog = XoPlus
    H = H_and_P[0]
    P = H_and_P[1]
    # print('H:', H, 'P', P)
    # print(H.shape, P.shape)
    from utils.init_utils import get_H_and_P
    H2, P2 = get_H_and_P(Xo, Xog, use=(1,))
    # print('H2:', H2, 'P2', P2)
    print('DIFF')
    # print(H.shape, P.shape)
    # print(H.shape, H2.shape, P.shape, P2.shape)
    print('H:', np.max(np.abs(H-H2)), 'P', np.max(np.abs(P-P2)))
    from matplotlib import pyplot as plt
    plt.plot(Xog, Xog*0, 'o')
    plt.plot(Xo.flatten(), Xo.flatten()*0, 'o')
    plt.show()