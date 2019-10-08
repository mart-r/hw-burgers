#!/usr/bin/env python3

from scipy.io import savemat
import os
import sys
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

def save(filename, **kwargs):
    if not filename.endswith('.mat'):
        filename += '.mat'
    print('saving to "%s/%s":'%(os.getcwd(),filename), '->', kwargs.keys())
    savemat(filename, kwargs)