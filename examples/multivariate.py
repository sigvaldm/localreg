from localreg import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Axes3D import has side effects, it enables using projection='3d' in add_subplot
import numpy as np

N = 500
degree=1

x = np.random.rand(N,2)
y = np.cos(2*np.pi*x[:,0])*(1-x[:,1])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

m = np.arange(0, 1.05, 0.05)
X, Y = np.meshgrid(m,m)
x0 = np.array([np.ravel(X), np.ravel(Y)]).T
z0 = localreg(x, y, x0, degree=degree, radius=0.2)
Z = z0.reshape(X.shape)

ax.plot_wireframe(X, Y, Z, rcount=10, ccount=10, color='green')
ax.plot3D(x[:,0], x[:,1], y, '.')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
