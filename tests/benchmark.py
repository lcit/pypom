#!/usr/bin/env python
""" Benchmarking of some POM functionalities written in 
    numpy and C++ to see which is faster.
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import time
from pypom import camera
from pypom import pom
import numpy as np
import time
from pypom import utils, core

def integral_array(a):
    S = a
    for i in range(a.ndim):
        S = S.cumsum(axis=i)
    return S

def integral_sum(a):
    return a[-1,-1]+a[0,0]-a[-1,0]-a[0,-1]   

A = np.float32(np.random.randn(100, 160, 320))
B = np.float32(np.random.randn(100, 160, 320)>0.5)

# ====================================================================
# ====================================================================
test = ("Sum", "numpy.sum(A)", "pom_core.sum(A)")
print("-----------Benchmarking "+test[0]+"--------------")
start = time.time()
for i in range(100):
    lAl = np.sum(A[i]) # --------->FASTER
print("Elapsed time for ({}): {:.6f}[s]".format(test[1], time.time()-start))

start = time.time()
for i in range(100):
    lAl = core.sum(A[i])
print("Elapsed time for ({}): {:.6f}[s]".format(test[2], time.time()-start))
# ====================================================================
# ====================================================================
test = ("Sum", "numpy.sum(B*A)", "pom_core.sum_mask(A,B)")
print("-----------Benchmarking "+test[0]+"--------------")
start = time.time()
for i in range(100):
    lAl = np.sum(B[i]*A[i]) # --------->SAME
print("Elapsed time for ({}): {:.6f}[s]".format(test[1], time.time()-start))

start = time.time()
for i in range(100):
    lAl = core.sum_mask(A[i], B[i]) # --------->SAME
print("Elapsed time for ({}): {:.6f}[s]".format(test[2], time.time()-start))
# ====================================================================
# ====================================================================
test = ("Integral image", "numpy~integral_image(A)", "pom_core.integral_image(A)")
print("-----------Benchmarking "+test[0]+"--------------")
start = time.time()
for i in range(100):
    Ai = integral_array(A[i]) 
print("Elapsed time for ({}): {:.6f}[s]".format(test[1], time.time()-start))

start = time.time()
for i in range(100):
    Ai = core.integral_image(A[i]) # --------->FASTER
print("Elapsed time for ({}): {:.6f}[s]".format(test[2], time.time()-start))
# ====================================================================
# ====================================================================
test = ("Integral sum", "numpy~integral_sum(A[..])", "pom_core.integral_sum(A,..)")
print("-----------Benchmarking "+test[0]+"--------------")
a = 1
start = time.time()
for _ in range(100):
    a += integral_sum(Ai[10:50, 20:70]) 
print("Elapsed time for ({}): {:.6f}[s]".format(test[1], time.time()-start))

start = time.time()
for _ in range(100):
    a += core.integral_sum(Ai, 10, 50, 20, 70)  # --------->FASTER
print("Elapsed time for ({}): {:.6f}[s]".format(test[2], time.time()-start))
# ====================================================================
# ====================================================================
test = ("Integral sum mask", "numpy~integral_sum(A*B)", "pom_core.integral_sum_mask(A,B)")
print("-----------Benchmarking "+test[0]+"--------------")
a = 1
start = time.time()
for i in range(100):
    a += core.integral_image(A[i]*B[i]) 
print("Elapsed time for ({}): {:.6f}[s]".format(test[1], time.time()-start))

start = time.time()
for i in range(100):
    a += core.integral_image_mask(A[i], B[i])   # --------->Slightly FASTER
print("Elapsed time for ({}): {:.6f}[s]".format(test[2], time.time()-start))
# ====================================================================
# ====================================================================
