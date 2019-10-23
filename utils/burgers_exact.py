#!/usr/bin/env python3
import sys
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")
import os.path
import mpmath as mp
import numpy as np
from scipy.special import iv
from utils.reprint import reprint
from scipy.io import savemat, loadmat

class BVGetter:
    __cacheFile = 'bessel_cache.mat'
    __cache = {}
    ccalc, ceps, cl, cu0, cinfty = None, None, None, None, None
    cbv = None

    def __init__(self):
        if os.path.isfile(self.__cacheFile):
            oldDps = mp.mp.dps
            mp.mp.dps = 800
            inD = loadmat(self.__cacheFile)
            for key in inD.keys():
                fkey = None
                try:
                    fkey = float(key)
                except ValueError:
                    continue
                if fkey is not None:
                    self.__cache[fkey] = [mp.mpf(el) for el in inD[key].flatten()]
            mp.mp.dps = oldDps

    def get_bv(self, calc, eps, l, u0, infty):
        R = float(u0 * l / (2 * np.pi * eps))
        if R in self.__cache:
            bv = self.__cache[R]
            existing = len(bv)
            if (existing < infty):
                bv += [calc.besseli(n, R) for n in range(existing, infty)]
                self.__cache[R] = bv
                self.save()
            elif infty < existing:
                bv = bv[:infty]
        else:
            bv = [calc.besseli(n, R) for n in range(infty)]
            self.__cache[R] = bv
            self.save()
        return np.array(bv)
    
    def save(self):
        oldDps = mp.mp.dps
        mp.mp.dps = 800
        saveDict = {}
        for key in self.__cache:
            saveDict[str(key)] = [str(el) for el in self.__cache[key]]
        savemat(self.__cacheFile, saveDict)
        mp.mp.dps = oldDps
    
bvGetter = BVGetter()

def get_bv(eps, l, u0, infty):
    return np.array([iv(n, u0 * l / (2 * np.pi * eps)) for n in range(infty)])

def get_bv_(calc, eps, l, u0, infty):
    if calc is np:
        return get_bv(eps, l, u0, infty)
    if calc is mp:
        return bvGetter.get_bv(calc, eps, l, u0, infty)

def exact_new_mp(xv, tv, eps, l=1, u0=1, infty=200):
    if hasattr(xv, '__len__'):
        print('xv  ... tv ... eps ... l ... u0 ... infty')
        print(xv[0], xv[-1], tv[0],tv[-1], eps, l, u0, infty)
    return exact_new_(mp, xv, tv, eps, l=l, u0=u0, infty=infty)

def exact_new_(calc, xv, tv, eps, l=1, u0=1, infty=200):
    print('DOING', infty, 'terms in sum')
    if calc is mp:
        if hasattr(xv, '__len__'):
            bxv = isinstance(xv[0], mp.mpf)
            xv = np.array([mp.mpf(el) for el in xv]).reshape(xv.shape)
        else:
            bxv = isinstance(xv, mp.mpf)
            xv = mp.mpf(xv)
        if hasattr(tv, '__len__'):
            btv = isinstance(tv[0], mp.mpf)
            tv = np.array([mp.mpf(el) for el in tv]).reshape(tv.shape)
        else:
            btv = isinstance(tv, mp.mpf)
            tv = mp.mpf(tv)
        beps = isinstance(eps, mp.mpf)
        if not bxv:
            print('xv not mpf:', xv)
        if not btv:
            print('tv not mpf:', tv)
        if not beps:
            print('eps not mpf:', eps)
            eps = mp.mpf(eps)
    if calc is np:
        eps = float(eps)
        u0 = float(u0)
        l = float(l)
    bv = get_bv_(calc, eps, l, u0, infty)
    xx, tt = np.meshgrid(xv, tv)
    downsum = calc.matrix(np.ones(np.shape(xx))).T * bv[0]
    upsum = calc.matrix(np.zeros(np.shape(tt))).T
    # print downsum.rows, downsum.cols
    # print upsum.rows, upsum.cols
    downsum_ = downsum
    upsum_ = upsum
    bAlreadyUp = False
    nr_up = 0
    bAlreadyDown = False
    nr_down = 0
    for n in range(1, infty):
        reprint('n=%d'%n)
        # print 'n=%d\r'%n,
        # sys.stdout.flush()
        if calc is np:
            # print eps, n, calc.pi, tv, l
            expval =  calc.matrix(np.exp(-eps * n**2 * calc.pi**2 * tv/l**2))  # depends on t
                             #remains np as mp doesn't support exponents nor sine for vectors
            xdepend = calc.matrix(np.sin(n * calc.pi * xv/l)).T  # depends on x
        else:
            val = -eps * n**2 * calc.pi**2 * tv/l**2
            if hasattr(tv, '__len__'):
                expval = calc.matrix([mp.exp(it) for it in val]).T
            else:
                expval = calc.exp(val)
            val = n * calc.pi * xv/l
            if hasattr(xv, '__len__'):
                xdepend = calc.matrix([mp.sin(it) for it in val])
            else:
                xdepend = calc.sin(val)
        # print np.shape(np.matrix(np.sin(n * np.pi * xv/l)).T)
        # print xdepend.rows, xdepend.cols
        # print expval.rows, expval.cols
        # print (xdepend.rows, xdepend.cols), (expval.rows, expval.cols)
        upadd = n * bv[n] * xdepend * expval  # depends on x andt
        upsum += upadd
        if calc is np:
            if hasattr(xv, '__len__'):
                xdepend = calc.matrix(np.cos(n * calc.pi * xv/l)).T  # depends on x
            else:
                xdepend = np.cos(n * calc.pi * xv/l)
        else:
            # val = n * calc.pi * xv/l # don't need as it's defined up already
            if hasattr(xv, '__len__'):
                xdepend = calc.matrix([mp.cos(it) for it in val])
            else:
                xdepend = calc.cos(val)
        downadd = 2 * bv[n] * xdepend * expval
        downsum += downadd
        if calc is np:
            diff1 = np.max(np.abs(upsum - upsum_))
            max1 = np.max(np.abs(upsum))
            diff2 = np.max(np.abs(downsum - downsum_))
            max2 = np.max(np.abs(upsum))
        else:
            diff1 = mp.norm(upsum - upsum_)
            max1 = mp.norm(upsum, p=1)
            diff2 = mp.norm(downsum - downsum_)
            max2 = mp.norm(downsum, p=1)
        diff1_ = diff1 if diff1 != 0 else mp.mp.eps
        diff2_ = diff2 if diff2 != 0 else mp.mp.eps
        # if diff1 < 10**-(mp.mp.dps - 5) and not bAlreadyUp:
        #     print 'max diff up', diff1, 'at', n, '(%d magn)'%mp.log10(max1/diff1_)
        #     nr_up += 1
        #     if nr_up > 100:
        #         bAlreadyUp = True
        # if diff2 < 10**-(mp.mp.dps - 5) and not bAlreadyDown:
        #     print 'max diff down', diff2, 'at', n, '(%d magn)'%mp.log10(max2/diff2_)
        #     nr_down += 1
        #     if nr_down > 100:
        #         bAlreadyDown = True
        downsum_ = downsum
        upsum_ = upsum
    print('')
    print('type', type(upsum), type(downsum))
    if calc is not np:
        upsum = upsum.tolist()
        downsum = downsum.tolist()
        print('diff in the very end', diff1, '(%d magn)'%mp.log10(max1/diff1_), diff2, '(%d magn)'%mp.log10(max2/diff2_))
        print(upsum[0][0],end='')
    else:
        print(upsum[0,0],end='')
    print('element (0,0)')
    retval = (4 * eps * np.pi / l * np.array(upsum) / np.array(downsum))
    print('max, min, mean, absmin')
    print(np.max(upsum), np.min(upsum), np.mean(upsum), np.min(np.abs(upsum)))
    print(np.max(downsum), np.min(downsum), np.mean(downsum), np.min(np.abs(downsum)))
    print('using mp ?', calc is mp)
    if calc is not np:
        print('in exact\n', mp.mp)
        return np.array(retval, dtype=float)
    return retval