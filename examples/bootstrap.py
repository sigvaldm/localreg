import numpy as np
import matplotlib.pyplot as plt
from localreg import *
from time import time

L = 2*np.pi
x = np.linspace(0, L, 5000)
x0 = np.linspace(0, L, 500)
yf = np.sin(x**2)
y = yf + 0.5*np.random.randn(*x.shape)

plt.plot(x, yf, label='$\\sin(x^2)$')
plt.plot(x, y, '+', markersize=0.2, color='black')

N = 200
yarr = np.zeros((N, len(x0)))

for n in range(N):
    ind = np.random.randint(0, len(x), len(x))
    xb = x[ind]
    yb = y[ind]
    yarr[n,:] = localreg(xb, yb, x0, degree=2, kernel=tricube, width=0.4)
    # plt.plot(x0, yarr[n,:], linewidth=0.5, color='black')

lower = np.percentile(yarr,  2.5, axis=0)
upper = np.percentile(yarr, 97.5, axis=0)
plt.fill_between(x0, lower, upper, color='gray', alpha=0.5)

y0 = localreg(x, y, x0, degree=2, kernel=tricube, width=0.4)
plt.plot(x0, y0, label='Local regression')

# ym = np.average(yarr, axis=0)
# plt.plot(x0, ym, '--', label='Bootstrapped local regression')

plt.title('Locally Weighted Polynomial Regression')
plt.xlabel('x')
plt.xlabel('y')
plt.legend()
plt.show()
