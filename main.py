import torch as torch
import numpy as np
import cupy as cp

import warnings
warnings.filterwarnings("ignore")

import cProfile, io, pstats
from pstats import SortKey

#tf.debugging.set_log_device_placement(True)
#tf.config.list_physical_devices(device_type = None)

#tf.add(1, 2).numpy()
#
#
#
#x_data = torch.tensor(data)
#
#print(x_data)

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

def gj_vectorized(A, b):
    A = np.array(A, float)
    b = np.array(b, float)
    n = len(A)

    b = np.reshape(b, (n, 1))
    Ab = np.concatenate((A, b), axis = 1)

    for k in range(n):
        row_to_top_np(Ab, k, n)

        # Division
        pivot = np.array(Ab[k, k])
        Ab[k] = np.divide(Ab[k], pivot)

        # Elimination
        for i in range(n):
            if i == k or Ab[i,k] == 0:
                continue
            factor = np.array(Ab[i,k])
            Ab[i, k:] = np.subtract(Ab[i, k:], np.multiply(Ab[k, k:], factor))

    return Ab[:, n:], Ab[:, :n]

def gj_cupy(A, b):
    A = cp.array(A, float)
    b = cp.array(b, float)
    n = len(A)

    b = cp.reshape(b, (n, 1))
    Ab = cp.concatenate((A, b), axis = 1)

    for k in range(n):
        row_to_top_cp(Ab, k, n)

        # Division
        pivot = cp.array(Ab[k, k])
        Ab[k] = cp.divide(Ab[k], pivot)

        # Elimination
        for i in range(n):
            if i == k or Ab[i,k] == 0:
                continue
            factor = cp.array(Ab[i,k])
            Ab[i, k:] = cp.subtract(Ab[i, k:], cp.multiply(Ab[k, k:], factor))

    return Ab[:, n:], Ab[:, :n]

def gj_torch(A, b):
    A = torch.tensor(A, dtype=torch.float)
    b = torch.tensor(b, dtype=torch.float)
    n = len(A)

    b = torch.reshape(b, (n, 1))
    Ab = torch.cat((A, b), dim = 1)

    for k in range(n):
        row_to_top_torch(Ab, k, n)

        # Division
        pivot = Ab[k, k].clone()
        Ab[k] = torch.div(Ab[k], pivot)

        # Elimination
        for i in range(n):
            if i == k or Ab[i,k] == 0:
                continue
            factor = Ab[i,k].clone()
            Ab[i, k:] = torch.sub(Ab[i, k:], torch.mul(Ab[k, k:], factor))

    return Ab[:, n:], Ab[:, :n]

def row_to_top_np(Ab, k, n):
    if np.fabs(Ab[k,k]) < 1.0e-12: 
        for i in range(k+1, n): 
            if np.fabs(Ab[i,k]) > np.fabs(Ab[k,k]): 
                temp = np.array(Ab[k])
                Ab[k] = Ab[i]
                Ab[i] = temp
                break

def row_to_top_cp(Ab, k, n):
    if cp.fabs(Ab[k,k]) < 1.0e-12: 
        for i in range(k+1, n): 
            if cp.fabs(Ab[i,k]) > cp.fabs(Ab[k,k]): 
                temp = cp.array(Ab[k])
                Ab[k] = Ab[i]
                Ab[i] = temp
                break

def row_to_top_torch(Ab, k, n):
    if torch.abs(Ab[k,k]) < 1.0e-12: 
        for i in range(k+1, n): 
            if torch.abs(Ab[i,k]) > torch.abs(Ab[k,k]): 
                temp = Ab[k].clone()
                Ab[k] = Ab[i]
                Ab[i] = temp
                break

A = [
    [0,2,0,1], 
    [2,2,3,2], 
    [4,-3,0,1], 
    [6,1,-6,-5]
    ]
b = [0,-2,-7,6]

x,a = gj(A,b)

print(f"Solution: {x}")

A = [
    [0,2,0,1], 
    [2,2,3,2], 
    [4,-3,0,1], 
    [6,1,-6,-5]
    ]
b = [0,-2,-7,6]

x,a = gj_cupy(A,b)

print(f"Solution: {x}")

A = [
    [0,2,0,1], 
    [2,2,3,2], 
    [4,-3,0,1], 
    [6,1,-6,-5]
    ]
b = [0,-2,-7,6]

x,a = gj_vectorized(A,b)

print(f"Solution: {x}")

A = [
    [0,2,0,1], 
    [2,2,3,2], 
    [4,-3,0,1], 
    [6,1,-6,-5]
    ]
b = [0,-2,-7,6]

x,a = gj_torch(A,b)

print(f"Solution: {x}")


def profile(_type="standard", n=1, rand=False, seed=23, log_to_file=False, m=4):
    '''
    _type (str): implementation to profile
        "torch" or "standard"
    n (int): number of iterations to profile
    rand (bool): whether or not to profile random matrices
    seed (int): seed to use when creating random matrices
    log_to_file (bool): whether or not to log to file
        save string: {_type}{n}{rand}{seed}.profile
    m (int): size of matrix and vector
        only used when rand=True
    '''
    if _type == "torch":
        func = gj_torch
    elif _type == "npvectorized":
        func = gj_vectorized
    elif _type == "cpvectorized":
        func = gj_cupy
    else:
        func = gj
    # Q: does python do truthy evaluation?!
    if rand == True:
        np.random.seed(seed=seed)
        A = np.random.rand(m, m)*12 - 6 # interval (-6, 6) as L.'s demo
        b = np.random.rand(m, 1)*12 - 6
        A = A.tolist()
        b = b.tolist()
    else:
        A = [
            [0,2,0,1], 
            [2,2,3,2], 
            [4,-3,0,1], 
            [6,1,-6,-5]
            ]
        b = [0,-2,-7,6]
    pr = cProfile.Profile()
    pr.enable()
    for i in range(n):
        x, a = func(A, b)
    pr.disable()
    if log_to_file == True:
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sortby)
        ps.print_stats()
        ps.print_callees()
        with open(f"{_type}_{n}_{m}{rand}{seed}.profile", "w") as f:
            print(s.getvalue(), file=f)

    # TODO: return the cumulative runtime

def compare():
    n = 100

    # profile(_type="torch", n=n, rand=True, seed=23, log_to_file=True, m=64)
    # profile(_type="cpvectorized", n=n, rand=True, seed=23, log_to_file=True, m=64)
    # profile(_type="npvectorized", n=n, rand=True, seed=23, log_to_file=True, m=64)
    # profile(_type="standard", n=n, rand=True, seed=23, log_to_file=True, m=64)
    
    # # profile(_type="torch", n=n, rand=True, seed=23, log_to_file=True, m=128)
    # profile(_type="cpvectorized", n=n, rand=True, seed=23, log_to_file=True, m=128)
    # profile(_type="npvectorized", n=n, rand=True, seed=23, log_to_file=True, m=128)
    # profile(_type="standard", n=n, rand=True, seed=23, log_to_file=True, m=128)
    
    # profile(_type="torch", n=n, rand=True, seed=23, log_to_file=True, m=256)
    # profile(_type="cpvectorized", n=n, rand=True, seed=23, log_to_file=True, m=256)
    profile(_type="npvectorized", n=n, rand=True, seed=23, log_to_file=True, m=256)
    profile(_type="standard", n=n, rand=True, seed=23, log_to_file=True, m=256)

    
    profile(_type="torch", n=n, rand=True, seed=23, log_to_file=True, m=1024)
    profile(_type="npvectorized", n=n, rand=True, seed=23, log_to_file=True, m=1024)

compare()

    
