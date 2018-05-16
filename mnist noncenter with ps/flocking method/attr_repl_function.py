from __future__ import unicode_literals
from sympy import *
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt


ATTRACTION = 0.5
REPULSION = 3.0
DIS = 0.001
SLOPE = REPULSION/DIS
NARROW = 2500000
NUM_ELEMENTS =10001



def linear_func(iter):
    a=np.array([0.0]*len(iter))
    for index, value in enumerate(iter):
        if np.abs(value)<DIS:
            a[index] = (SLOPE*np.abs(value)- REPULSION + ATTRACTION)
        else:
            a[index] = 0.5
    return a

x = symbols('x')
func = (ATTRACTION - REPULSION*exp(-NARROW*x**2))

X = np.linspace(-0.01, 0.01, NUM_ELEMENTS)
f = lambdify(x, func, "numpy")
Y = f(X)


linear_Y = linear_func(X)
print(linear_Y)
#print(Y)
plt.plot(X, Y, 'b')
plt.plot(X, linear_Y, 'r')
plt.xlim(-0.01, 0.01)
# plt.plot(X, [0.0]*NUM_ELEMENTS, 'k-')
plt.xlabel(r'$\theta_{i,t}-\bar{\theta}_{i,t}$')
plt.ylabel(r'$f(\theta_{i,t}-\bar{\theta}_{i,t})$')
plt.legend([r'exponential form', r'piecewise linear form'])
# plt.legend(['original flocking function', 'linear flocking function'])
plt.grid()
plt.show()