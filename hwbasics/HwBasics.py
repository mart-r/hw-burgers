#!/usr/bin/env python3

import numpy as np

def Pn_nu(J, n, Xg, Xc):
    M = 2**J
    N = 2*M
    f = np.math.factorial(n)
    
    P = np.zeros((N,N))
    for l in range(N):
        P[0,l]=(Xc[l]-Xg[0])**n/f

    # print (P.shape)
    # print (Xc.shape)
    # print (Xg.shape)
    for j in range(J+1):
        m = 2**j
        for k in range(m):
            i = m + k 
            nu = int(M/m)
            ksi1 = Xg[2 * k * nu]
            ksi2 = Xg[(2 * k + 1) * nu]
            ksi3 = Xg[2 * (k + 1) * nu]
            ci = (ksi2-ksi1)/(ksi3-ksi2)
            for l in range(N):
                x = Xc[l]
                if (x >= ksi1 and x < ksi2):
                    P[i,l] = (x-ksi1)**n/f
                elif (x >= ksi2 and x < ksi3):
                    P[i,l] = ((x-ksi1)**n - (1+ci)*(x-ksi2)**n)/f
                elif (x>=ksi3):
                    P[i,l] = ((x-ksi1)**n - (1+ci)*(x-ksi2)**n + ci*(x-ksi3)**n)/f
                # print(i,l,P[i,l])
    return P

def  Pnx_nu(J, n, X, Xg):
    M = 2**J
    N = 2*M
    f = np.math.factorial(n)
    
    PX = np.zeros((N,1))

    PX[0]=(X-Xg[0])**n/f
    for j in range(J+1):
        m = 2**j
        for k in range(m):
            i = m + k
            nu = int(M/m);
            ksi1 = Xg[2 * k * nu]
            ksi2 = Xg[(2 * k + 1) * nu]
            ksi3 = Xg[2 * (k + 1) * nu]
            ci = (ksi2-ksi1)/(ksi3-ksi2)
            for l in range(N):
                if (X >= ksi1 and X < ksi2):
                    PX[i] = (X-ksi1)**n/f
                elif (X >= ksi2 and X < ksi3): 
                    PX[i]=((X-ksi1)**n-(1+ci)*(X-ksi2)**n)/f
                elif (X >= ksi3):
                        PX[i]=((X-ksi1)**n-(1+ci)*(X-ksi2)**n+ci*(X-ksi3)**n)/f
    return PX



def Hm_nu(X):
    N = len(X)-1
    J = np.log2(N/2)
    if (np.floor(J) != J):
        raise Exception('Need the number of grid points to be a power of 2 + 1, found %d', N+1)
    J = int(J)
    M = N/2
    H = np.zeros((N,)*2)
    H[0,:] += 1
    # print('N=%d\tJ=%d\tM=%d'%(N,J,M))

    for j in range(J+1):
        m = 2**j
        # print('m=%d\n'%m)
        for k in range(m):
            i = m + k
            nu = int(M/m)
            # print('k=%d_i=%d_nu=%g\n'%(k, i, nu), end='')
            # print('->', 2 * k * nu)
            ksi1 = X[2 * k * nu]
            ksi2 = X[(2 * k + 1) * nu]
            ksi3 = X[2 * (k + 1) * nu]
            ci = (ksi2-ksi1)/(ksi3-ksi2)
            # print('xi1=%12g\txi2=%12g\txi3=%12g'%(ksi1, ksi2, ksi3), end='')
            # print('\tci=%g\n'%ci, end='')
            for l in range(N):
                x = X[l]
                if (x >= ksi1 and x < ksi2):
                    H[i,l] = 1
                elif (x >= ksi2 and x < ksi3):
                    H[i,l] = -ci
            # print('_: %8g', H[i,l])
    return H

