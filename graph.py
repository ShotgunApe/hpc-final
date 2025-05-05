import matplotlib.pyplot as plt
import numpy as np

sizes = [2, 4, 8, 32, 64, 128]# log base 2
imps = ["standard", "npvectorized", "cpvectorized", "torch"]
standard = [0.002, 0.008, 0.026, 0.808, 5.171, 36.510]
npvectorized = [0.003, 0.007, 0.017, 0.246, 0.983, 3.947, 16.306]
cpvectorized = [0.131, 0.386, 1.488, 23.965, 104.575, 411.239]
torch = [0.017, 0.044, 0.230, 2.657, 10.757, 42.930, 172.023]

plt.plot(sizes, standard)
plt.plot(sizes + [256, 1024], npvectorized)
plt.plot(sizes, cpvectorized)
plt.plot(sizes + [256, 1024], torch)
plt.yscale('log')
plt.xscale('log')
plt.legend(imps)
plt.xlabel("log(m)")
plt.ylabel("Runtime (s)")
plt.savefig("log.png")

