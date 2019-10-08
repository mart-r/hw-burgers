#!/usr/bin/env python3

import sys

def reprint(s):
    print(str(s) + '\r',end='')
    sys.stdout.flush()


if __name__ == '__main__':
    # testing
    import time
    for i in range(10):
        reprint(i)
        time.sleep(.5)
    print('')