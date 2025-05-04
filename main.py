import tensorflow as tf
import torch as torch
import numpy as np

import warnings
warnings.filterwarnings("ignore")

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
        print(f"b after swap {k}: {b}")
        # Division
        pivot = A[k,k]
        for j in range(k,n): 
            A[k,j] /= pivot
        b[k] /= pivot
        print(f"b after division {k}: {b}")
        #Elimination
        for i in range(n): 
            if i == k or A[i,k] == 0: 
                continue
            factor = A[i,k]
            for j in range(k,n): 
                A[i,j] -= factor*A[k,j]
            b[i] -= factor*b[k]

        print(f"b after elimination {k}: {b}")
    
    return b, A

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

x,a = gj_torch(A,b)

print(f"Solution: {x}")