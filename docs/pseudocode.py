import tensorflow as tf
import torch as torch
import numpy as np


def gj(A,b): 
    A = np.array(A, float)
    b = np.array(b, float)
    n = len(b)
    
    for k in range(n): 
        if np.fabs(A[k,k]) < 1.0e-12: 
            for i in range(k+1, n): 
                if np.fabs(A[i,k]) > np.fabs(A[k,k]): 
                    for j in range(k,n): 
                        A[k,j],A[i,j] = A[i,j],A[k,j]
                    b[k],b[i] = b[i],b[k]
                    break
        # Division
        pivot = A[k,k]
        for j in range(k,n): 
            A[k,j] /= pivot
        b[k] /= pivot
        #Elimination
        for i in range(n): 
            if i == k or A[i,k] == 0: 
                continue
            factor = A[i,k]
            for j in range(k,n): 
                A[i,j] -= factor*A[k,j]
            b[i] -= factor*b[k]
    
    return b, A
