#!/usr/bin/env python3

import sys, getopt
sys.path.append('/home/mart/Documents/KybI/2019/python/NewPython2019Oct')
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")
from tests.test_helpers import save

def find_X_and_save(J, a, outputFile):
    from utils.nonuniform_grid import nonuniform_grid
    X, Xg = nonuniform_grid(J, a)
    save(outputFile, X=X, Xg=Xg)


def parse_arguments(argv):
    __usage = argv[0] + ' -j <J> -a <a> -o <outputfile>'
    argv = argv[1:]

    J = None
    a = None
    outputfile = None
    try:
        opts, args = getopt.getopt(argv,"hj:a:o:",["ofile="])
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
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    if J is None or outputfile is None or a is None:
        print(__usage)
        sys.exit(2)
    print('J is "%d"'%J)
    print('a is "%d"'%a)
    print('Output file is "%s"'%outputfile)
    return J, a, outputfile

if __name__ == "__main__":
   J, a, outputFile = parse_arguments(sys.argv)
   find_X_and_save(J, a, outputFile)