#!/usr/bin/env python3

from numpy import pi # in case you want pi (i.e in nu evaluation)
import sys
# my "package"
sys.path.append('/home/mart/Documents/KybI/2019/python/NewPython2019Oct')
from higherlevel.hw_euler_burgers import hw_euler_burgers_newest as solver

def parse_nu(arg):
    try:        
        nu = float(arg)
    except ValueError:
        # try eval
        nu = eval(arg)
    return nu


def parse_arguments(args):
    _usage1 = 'arguments:  J   nu     tf     bHO    nua'
    _usage2 = 'arg types: INT FLOAT FLOAT BOOL/INT FLOAT'
    _usage = _usage1 + '\n' + _usage2
    # defaults
    tf = 1/2
    bHO = False
    nua = 1
    nargs = len(args)
    if nargs < 2:
        print(_usage)
        raise Exception("Need at least 2 arguments! (J and nu)")
    try:
        J = int(args[0])
    except ValueError:
        print(_usage)
        raise Exception("Value for J needs to be an integer!")
    try:
        nu = parse_nu(args[1])
    except Exception:
        print(_usage)
        raise Exception("Value for nu needs to be a float!")
    if nargs > 2:
        try:        
            tf = float(args[2])
        except ValueError:
            print(_usage)
            raise Exception("Value for tf needs to be a float!")
    if nargs > 3:
        arg = ''
        try:
            arg = int(args[3])
        except ValueError:
            arg = args[3]
        try:        
            bHO = bool(arg)
        except ValueError:
            print(_usage)
            raise Exception("Value for bHO needs to be a boolean or an integer!")
    if nargs > 4:
        try:        
            nua = float(args[4])
        except ValueError:
            print(_usage)
            raise Exception("Value for nua needs to be a float!")
    return J, nu, tf, bHO, nua


if __name__ == '__main__':
    summax = 200 # not needed here
    args = sys.argv[1:]
    u01 = 1 # not changing, for now at least
    L = 1   # not changing, for now at least
    J, nu, tf, bHO, nua = parse_arguments(args)
    print('J, nu, tf, bHO, nua')
    print(J, nu, tf, bHO, nua)
    solver(J, nu, tf, summax, u01, L, bHO, nua)
