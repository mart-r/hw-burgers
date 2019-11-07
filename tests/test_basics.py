#!/usr/bin/env python3

import sys, getopt
print (sys.path[0])
sys.path.append('/home/mart/Documents/KybI/2019/python/hw-burgers')
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")
from tests.test_helpers import save

def find_P_and_save(J, n, a, outputFile):
    from utils.nonuniform_grid import nonuniform_grid
    X, Xg = nonuniform_grid(J, a)
    from hwbasics.HwBasics import Pn_nu
    P = Pn_nu(J, n, Xg, X)
    save(outputFile, P=P)

def find_Px_and_save(J, n, a, outputFile):
    from utils.nonuniform_grid import nonuniform_grid
    Xg = nonuniform_grid(J, a)[1]
    from hwbasics.HwBasics import Pnx_nu
    Px = Pnx_nu(J, n, 1, Xg)
    save(outputFile, Px=Px)


def find_H_and_save(J, n, a, outputFile):
    from utils.nonuniform_grid import nonuniform_grid
    Xg = nonuniform_grid(J, a)[1]
    from hwbasics.HwBasics import Hm_nu
    H = Hm_nu(Xg)
    save(outputFile, H=H)


def parse_arguments(argv):
    __usage = argv[0] + ' -j <J> -a <a> -n <n> -o <outputfile>'
    argv = argv[1:]

    J = None
    a = None
    n = None
    outputfile = None
    doP = False
    doPx = False
    try:
        opts = getopt.getopt(argv,"hj:a:n:o:px",["ofile="])[0]
    except getopt.GetoptError:
        print(__usage)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(__usage)
            sys.exit()
        elif opt in ("-j", "--j"):
            try:
                J = int(arg)
            except ValueError:
                print('Need to pass number for J!')
                sys.exit(2)
        elif opt == '-a':
            try:
                a = float(arg)
            except ValueError:
                print('Need to pass float for a!')
                sys.exit(2)
        elif opt == '-n':
            try:
                n = int(arg)
            except ValueError:
                print('Nueed to pass number for n!')
                sys.exit(2)
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt == '-p':
            doP = True
        elif opt == '-x':
            doPx = True
    if J is None or a is None or n is None or outputfile is None:
        print(__usage)
        sys.exit(2)
    print('J=%d'%J)
    print('a=%f'%a)
    print('n=%d'%n)
    print('Output file is "%s"'%outputfile)
    return J, a, n, outputfile, doP, doPx

if __name__ == "__main__":
    J, a, n, outputFile, doP, doPx = parse_arguments(sys.argv)
    if doP:
        find_P_and_save(J, n, a, outputFile)
    elif doPx:
        find_Px_and_save(J, n, a, outputFile)
    else:
        find_H_and_save(J, n, a, outputFile)
