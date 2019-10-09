#!/usr/bin/env python3

from numpy import pi # in case you want pi (i.e in nu evaluation)
import sys
# my "package"
sys.path.append('/home/mart/Documents/KybI/2019/python/NewPython2019Oct')
from higherlevel.hw_euler_burgers import hw_euler_burgers_newest as solver, saver, plot_results
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

def parse_nu(arg):
    vals = arg.split(',')
    nu = []
    for curVal in vals:
        cnu = parse_nu_2(curVal)
        nu.append(cnu)
    return nu


def parse_nu_2(arg):
    try:        
        nu = float(arg)
    except ValueError:
        # try eval
        nu = eval(arg)
    return nu


def parse_bool(argIn):
    arg = ''
    try:
        arg = int(argIn)
    except ValueError:
        arg = argIn
    parsedBool = bool(arg)
    return parsedBool


def parse_arguments(args):
    _usage1 = 'arguments:    J            nu          tf     bHO    nua'
    _usage2 = 'arg types: INT:INT FLOAT:FLOAT:FLOAT FLOAT BOOL/INT FLOAT'
    _usage = _usage1 + '\n' + _usage2
    # defaults
    tf = 1/2
    bHO = False
    nua = 1
    bFindExact = True
    # parse
    nargs = len(args)
    if nargs < 2:
        print(_usage)
        raise Exception("Need at least 2 arguments! (J and nu)")
    try:
        vals = args[0].split(':')
        if (len(vals) != 2):
            raise ValueError
        J1 = int(vals[0])
        J2 = int(vals[1])
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
        try:
            bHO = parse_bool(args[3])
        except Exception:
            print(_usage)
            raise Exception("Value for bHO needs to be a boolean or an integer!")
    if nargs > 4:
        try:        
            nua = float(args[4])
        except ValueError:
            print(_usage)
            raise Exception("Value for nua needs to be a float!")
    if nargs > 5:
        try:        
            bFindExact = float(args[5])
        except ValueError:
            bFindExact = True 
    J = range(J1, J2+1)
    return J, nu, tf, bHO, nua, bFindExact


if __name__ == '__main__':
    summax = 200 # not needed here
    args = sys.argv[1:]
    u01 = 1 # not changing, for now at least
    L = 1   # not changing, for now at least
    JAll, nuAll, tf, bHO, nua, bFindExact = parse_arguments(args)
    for J in JAll:
        for nu in nuAll:
            print('J, nu, tf, bHO, nua, bFindExact')
            print(J, nu, tf, bHO, nua, bFindExact)
            X, T, U, Ue = solver(J, nu, tf, summax, u01, L, bHO, nua, bFindExact)
            fname = 'HW_Burgers_AS_J=%d_nu=%f_tf=%d_nua=%f'%(J, nu, tf, nua)
            if bHO:
                fname += "_HO"
            fname += ".mat"
            plot_results(X, T,U ,Ue, False)
            saver(fname, X, T, U, Ue)
