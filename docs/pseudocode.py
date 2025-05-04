import tensorflow as tf
import torch as torch
import numpy as np

#tf.debugging.set_log_device_placement(True)
#tf.config.list_physical_devices(device_type = None)

#tf.add(1, 2).numpy()
#
#def g_j_tf(A):
#    tf_A = tf.constant(A)
#    
#
#def g_j_torch(A):
#    torch_A = torch.tensor(A)
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


A = np.array([
    [0,2,0,1], 
    [2,2,3,2], 
    [4,-3,0,1], 
    [6,1,-6,-5]
    ])
b = np.array([0,-2,-7,6])

x,a = gj(A,b)

print(f"Solution: {x}")