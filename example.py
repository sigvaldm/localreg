import numpy as np
import matplotlib.pyplot as plt
from localreg import *
from time import time

x = np.linspace(0, 2*np.pi, 5000)
yf = np.sin(x**2)
y = yf + 0.5*np.random.randn(*x.shape)

# y0 = localreg(x, y, degree=4, kernel=tricube, width=0.4)
y0 = localreg(x, y, degree=4, kernel=gaussian, width=0.1)

plt.plot(x, yf, label='$\\sin(x^2)$')
plt.plot(x, y, '+', markersize=0.2, color='black')
plt.plot(x, y0, label='Local regression')
plt.title('Locally Weighted Polynomial Regression')
plt.xlabel('x')
plt.xlabel('y')
plt.legend()
plt.show()
