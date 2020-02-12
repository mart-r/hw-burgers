#!/usr/bin/env python3

import numpy as np
import sys
# integration
from scipy.integrate import ode
from scipy.interpolate import interp1d

from utils.nonuniform_grid import nonuniform_grid

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

from enum import Enum

class AdaptiveGridType(Enum):
    STATIONARY = 0
    DERIV_NU_PLUS_BORDERS = 1 # DEFAULT
    STATIONARY_MIDDLE = 2


class AdaptiveGrid:

    def __init__(self, J: int, gridType: AdaptiveGridType, mid: float, borders: int, **kwargs):
        # general parameters
        self.J = J
        self.M2 = 2 * 2**J
        self.gridType = gridType
        self.mid = mid # not used for DERIV_NU_PLUS_BORDERS
        if "bc" in kwargs: # boundary conditions: [u(0, t), u(1,t)]
            self.bc = kwargs["bc"]
        else:
            self.bc = [0, 0]
        # grid specific parameters
        if gridType is AdaptiveGridType.DERIV_NU_PLUS_BORDERS:
            if "a" not in kwargs:
                raise ValueError("Need to specify keyword argument 'a'")
            if "hw" not in kwargs: # halfwidth aroudn the maxima/minima
                raise ValueError("Need to specify keyword argument 'hw'")
            if "Nper" not in kwargs: # number of points per half width, defaults to 'int(M2/4) + 1'
                kwargs["Nper"] = int(self.M2/4) + 1 # default
            if "onlyMin" not in kwargs: # whether or not to use only the maximal value (i.e ignore minimal)
                kwargs["onlyMin"] = False # default
            if "onlyMax" not in kwargs: # whether or not to use only the maximal value (i.e ignore minimal)
                kwargs["onlyMax"] = False # default
            if "nuBorders" not in kwargs: # whether or not to create nonuniform borders as well
                kwargs["nuBorders"] = False
            self.a = kwargs["a"]
            if not isinstance(self.a, float):
                raise ValueError("Need to specify a floating point number for a, got %s"%str(self.a))
            if self.a < 0:
                raise ValueError("Nonuniform parameter 'a' cannot be lower than 0, got %f"%self.a)
            self.halfWidth = kwargs["hw"]
            if not isinstance(self.halfWidth, float):
                raise ValueError("Need to specify a floating point number for a, got %s"%str(self.a))
            if self.halfWidth * 2 > 1:
                raise ValueError("Halfwidth cannot be greater than half of the domain, got %f"%self.halfWidth)
            self.Nper = kwargs["Nper"]
            if not isinstance(self.Nper, int):
                raise ValueError("Expected integer for Nper, got %s"%str(self.Nper))
            if self.Nper < 0 or self.Nper * 2 > self.M2:
                raise ValueError("Expected Nper to between 0 and %d, got %d"%(self.M2, self.Nper))
            self.onlyMin = kwargs["onlyMin"]
            if not isinstance(self.onlyMin, bool):
                raise ValueError("Expected onlyMin to be a boolean, got %s"%str(self.onlyMin))
            self.onlyMax = kwargs["onlyMax"]
            if not isinstance(self.onlyMax, bool):
                raise ValueError("Expected onlyMax to be a boolean, got %s"%str(self.onlyMax))
            if self.onlyMin and self.onlyMax:
                raise ValueError("Cannot have both only min and only max!")
            self.nuBorders = kwargs["nuBorders"]
            if not isinstance(self.nuBorders, bool):
                raise ValueError("Expected nuBorders to be a boolean, got %s"%str(self.nuBorders))
        elif gridType is AdaptiveGridType.STATIONARY:
            if "a" not in kwargs:
                kwargs["a"] = 1
            self.a = kwargs["a"]
            if self.a < 0:
                raise ValueError("Nonuniform parameter 'a' cannot be lower than 0, got %f"%self.a)
            self.Xg = nonuniform_grid(self.J, self.a)[1]
        elif gridType is AdaptiveGridType.STATIONARY_MIDDLE:
            if "gap" not in kwargs:
                kwargs["gap"] = .1
            if "borders" not in kwargs:
                kwargs["borders"] = 1
            self.gap = kwargs["gap"]
            if self.gap < 0 or self.gap > .5:
                raise ValueError("Parameter 'gap' cannot be lower than 0 or bigger than 1/2, got %f"%self.gap)
            self.borders = kwargs["borders"]
            if self.borders < 1 or self.borders > self.M2/2:
                raise ValueError("Parameter 'borders' cannot be lower than 1 or bigger than J^2, got %d"%self.borders)
            Xg = nonuniform_grid(self.J, 1)[1] # uniform, [0, 1]
            mid = Xg[borders:-borders] # correct number of elements
            mid -= mid[0]           # from [0, 1 - borders/M2]
            mid /= mid[-1]          # from [0, 1], correct number of elmenets
            mid *= 1 - 2 * self.gap # from [0, 1-2*gap]
            mid += self.gap         # from [gap, 1 - gap]
            left = np.linspace(0, self.gap, borders + 1)[:-1]
            right = np.linspace(1 - self.gap, 1, borders + 1)[1:]
            self.Xg = np.hstack((left, mid, right))
        else:
            raise ValueError("Other grids have not yet been implemented!")
    
    def get_grid(self, Xcur, weights):
        weights = weights.flatten()
        Xcur = Xcur.flatten()
        if (weights.shape != Xcur.shape):
            raise ValueError("Number of grid points must be the same, got %s and %s"%(str(weights.shape), str(Xcur.shape)))
        if self.gridType is AdaptiveGridType.DERIV_NU_PLUS_BORDERS:
            Xg = self.__get_deriv_nu_plus_borders(Xcur, weights)
        elif self.gridType is AdaptiveGridType.STATIONARY or \
                self.gridType is AdaptiveGridType.STATIONARY_MIDDLE:
            Xg = self.Xg
        else:
            raise ValueError("Grids other than DERIV_NU_PLUS_BORDERS have not yet been implemented")
        X = (Xg[1:]+Xg[:-1])/2
        return Xg, X.reshape(1, self.M2)

    def __get_deriv_nu_plus_borders(self, Xcur, weights):
        # find
        if self.a == 1:
            nuPart = np.linspace(0, 1, self.Nper + 1) * self.halfWidth
        else:
            nuPart = (self.a**np.arange(self.Nper + 1) - 1)/(self.a**self.Nper - 1) * self.halfWidth
        if self.onlyMin:
            iMin = np.argmin(weights)
        elif self.onlyMax:
            iMax = np.argmax(weights)
        if not self.onlyMin and not self.onlyMax:
            iMin = np.argmin(weights)
            nuPartMid = nuPart[-int(self.Nper/2):]
            iMax = np.argmax(weights)

            maxPart = np.hstack((nuPart[:-1] + Xcur[iMax] - self.halfWidth, Xcur[iMax] + self.halfWidth - nuPartMid[::-1]))
            minPart = np.hstack((nuPartMid[:-1] + Xcur[iMin] - self.halfWidth, Xcur[iMin] + self.halfWidth - nuPart[::-1]))

            if iMax < iMin:
                overlap = np.sum(minPart < maxPart[-1])
                while overlap > 0:
                    minPart = minPart[1:]
                    maxPart = maxPart[:-1]
                    overlap = np.sum(minPart < maxPart[-1]) 
            else:
                raise ValueError("Not implemented case where xMin < xMax")
            mid = np.hstack((maxPart, minPart))
        else:
            x = Xcur[iMin] if self.onlyMin else Xcur[iMax]
            lp = nuPart[:-1] + x - self.halfWidth
            rp = x + self.halfWidth - nuPart[::-1]
            mid = np.hstack((lp, rp))
        mid = mid[np.sum(mid < 0):] # remove numbers less than 0
        nrOfGreaterThan = np.sum(mid > 1) # remove numbers greater than 1
        if nrOfGreaterThan > 0:
            mid = mid[:-nrOfGreaterThan]

        left, right = mid[0], mid[-1]
        nrl, nrr = self.get_smart_left_right(self.M2 - len(mid) + 1, left, 1 - right)
        if self.nuBorders:
            a = (self.a + 1)/2 # below 0 -> tighter in the end
            lXg = (a**np.arange(nrl) - 1)/(a**nrl - 1) * left
            a = 2 - a
            rXg = (a**np.arange(1, nrr + 1) - 1)/(a**nrr - 1) * (1 - right) + right
        else:
            lXg = np.linspace(0, left, nrl + 1)[:-1]
            rXg =  np.linspace(right, 1, nrr + 1)[1:]
        
        Xg = np.hstack((lXg, mid, rXg))
        return Xg

    def get_smart_left_right(self, totalBorders, left, right):
        nrl = totalBorders * left /(left  + right)
        nrr = totalBorders * right /(left + right)
        nrl, nrr = int(nrl), int(nrr)
        if nrl + nrr < totalBorders:
            if nrl/left < nrr/right:
                nrl, nrr = nrl + 1, nrr
            else:
                nrl, nrr = nrl, nrr + 1
        return nrl, nrr
