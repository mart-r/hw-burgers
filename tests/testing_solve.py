#!/usr/bin/env python3

import sys, getopt
print (sys.path[0])
sys.path.append('/home/mart/Documents/KybI/2019/python/NewPython2019Oct')
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")
import numpy as np
import time


def test_0(Jx=3, Jy=2):
    Mx = 2**Jx
    Mx2 = 2 * Mx
    My = 2**Jy
    My2 = 2 * My
    h1 = np.arange(Mx2**2*My2**2).reshape(My2, Mx2, My2, Mx2, order='F') # (order doesn't really matter)
    h2 = h1.reshape(My2 * Mx2, My2 * Mx2, order='F') # order DOES matter here
    for i in range(My2):
        for j in range(Mx2):
            for k in range(My2):
                for l in range(Mx2):
                    where = np.where(h2 == h1[i,j,k,l])
                    to = (where[0][0], where[1][0])
                    to2 = (j * My2 + i,l * My2 + k)
                    if (to != to2):
                        print(i,j,k,l, "->", to, 'vs', to2)


def test_1(Jx=2, Jy=3, tol=1e-13, timesToRepeat=10):
    Mx2 = 2 * 2**Jx
    My2 = 2 * 2**Jy
    A = np.random.random((My2,My2))
    B = np.random.random((Mx2,Mx2))
    f = np.random.random((My2,Mx2))
    # kron solution
    start = time.time()
    for repeat in range(timesToRepeat):
        vecsol1 = np.linalg.solve(np.kron(B.T, A), f.reshape(Mx2 * My2, 1, order='F'))
        sol1 = vecsol1.reshape(My2, Mx2, order='F')
    took1 = time.time() - start

    # non-kron sol
    start = time.time()
    for repeat in range(timesToRepeat):
        mat2 = np.zeros((My2, Mx2, My2, Mx2))
        for i in range(My2):
            for j in range(Mx2):
                for k in range(My2):
                    for l in range(Mx2):
                        mat2[i,j,k,l] = A[i,k] * B[l,j]
        mat2 = mat2.reshape(My2 * Mx2, My2 * Mx2, order='F')
        vecsol2 = np.linalg.solve(mat2, f.reshape(Mx2 * My2, 1, order='F'))
        sol2 = vecsol2.reshape(My2, Mx2, order='F')
    took2 = time.time() - start

    # non-kron sol 2
    start = time.time()
    for repeat in range(timesToRepeat):
        mat3 = np.zeros((Mx2 * My2, )*2)
        for i in range(My2):
            for j in range(Mx2):
                matI = j * My2 + i
                for k in range(My2):
                    for l in range(Mx2):
                        matJ = l * My2 + k
                        mat3[matI, matJ] = A[i,k] * B[l,j]
        vecsol3 = np.linalg.solve(mat3, f.reshape(Mx2 * My2, 1, order='F'))
        sol3 = vecsol3.reshape(My2, Mx2, order='F')
    took3 = time.time() - start

    # accuracy
    diff11 = np.max(np.abs(A @ sol1 @ B - f))
    diff12 = np.max(np.abs(np.dot(A, np.dot(sol1, B)) - f))
    diff21 = np.max(np.abs(A @ sol2 @ B - f))
    diff22 = np.max(np.abs(np.dot(A, np.dot(sol2, B)) - f))
    diff31 = np.max(np.abs(A @ sol3 @ B - f))
    diff32 = np.max(np.abs(np.dot(A, np.dot(sol3, B)) - f))

    if (diff11 > tol) or (diff12 > tol) or (diff21 > tol) or (diff22 > tol) or (diff31 > tol) or (diff32 > tol):
        print('DIFFERENCE TOO BIG:', diff11, diff12, diff21, diff22, diff31, diff32)
    print('Took %gs for kronecker and %gs for other, %gs for third (on average per %d)'%(took1/timesToRepeat, took2/timesToRepeat, took3/timesToRepeat, timesToRepeat))


def test_2(Jx=2, Jy=2, timesToRepeat=10):
    Mx2 = 2 * 2**Jx
    My2 = 2 * 2**Jy
    A = np.random.random((My2,My2))
    B = np.random.random((Mx2,Mx2))
    C = np.random.random((Mx2,Mx2))
    D = np.random.random((My2,My2))
    U1 = np.random.random((My2,Mx2))
    U2 = np.random.random((My2,Mx2))
    Iy = np.eye(My2)
    Ix = np.eye(Mx2)
    f = np.random.random((My2,Mx2))

    # kron approach 1                  - F type reshape, take first row
    start = time.time()
    for repeat in range(timesToRepeat):
        mat1 = np.kron(B.T, A) + \
                np.diag(U1.reshape(1, My2*Mx2, order='F')[0]) @ np.kron(C.T, Iy) + \
                np.diag(U2.reshape(1, My2*Mx2, order='F')[0]) @ np.kron(Ix.T, D)
        solvec1 = np.linalg.solve(mat1, f.reshape(My2*Mx2, 1, order='F'))
        sol1 = solvec1.reshape(My2, Mx2, order='F')
    took1 = (time.time() - start)/timesToRepeat
    diff1 = np.max(np.abs(A @ sol1 @ B + U1 * (Iy @ sol1 @ C) + U2 * (D @ sol1 @ Ix) - f))
    print('took:', took1, 'diff', diff1)

    # kron approach 2                  - F type reshape, take first row
    start = time.time()
    for repeat in range(timesToRepeat):
        mat2 = np.kron(B.T, A) + \
                np.kron(C.T, Iy) @ np.diag(U1.reshape(1, My2*Mx2, order='F')[0]) + \
                np.kron(Ix.T, D) @ np.diag(U2.reshape(1, My2*Mx2, order='F')[0]) 
        solvec2 = np.linalg.solve(mat2, f.reshape(My2*Mx2, 1, order='F'))
        sol2 = solvec1.reshape(My2, Mx2, order='F')
    took2 = (time.time() - start)/timesToRepeat
    diff2 = np.max(np.abs(A @ sol2 @ B + U1 * (Iy @ sol2 @ C) + U2 * (D @ sol2 @ Ix) - f))
    print('took:', took2, 'diff', diff2)
    
    # non-kron approach
    start = time.time()
    for repeat in range(timesToRepeat):
        mat3 = np.zeros((My2, Mx2, My2, Mx2))
        for i in range(My2):
            for j in range(Mx2):
                for k in range(My2):
                    for l in range(Mx2):
                        mat3[i,j,k,l] = A[i,k] * B[l,j] + U1[i,j] * Iy[i,k] * C[l,j] + U2[i,j] * D[i,k] * Ix[l,j]
        mat3 = mat3.reshape(My2 * Mx2, My2 * Mx2, order='F')
        solvec3 = np.linalg.solve(mat3, f.reshape(Mx2 * My2, 1, order='F'))
        sol3 = solvec3.reshape(My2, Mx2, order='F')
    took3 = (time.time() - start)/timesToRepeat
    diff3 = np.max(np.abs(A @ sol3 @ B + U1 * (Iy @ sol3 @ C) + U2 * (D @ sol3 @ Ix) - f))
    print('took:', took3, 'diff', diff3)


if __name__ == '__main__':
    test_0()
    test_1(3,2)
    test_2(2,3)