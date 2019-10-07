#!/usr/bin/env python3
from scipy.io import savemat
import os

def save(filename, **kwargs):
    if not filename.endswith('.mat'):
        filename += '.mat'
    print('saving to "%s/%s":'%(os.getcwd(),filename), '->', kwargs.keys())
    savemat(filename, kwargs)