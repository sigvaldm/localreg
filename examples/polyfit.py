from localreg import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Axes3D import has side effects, it enables using projection='3d' in add_subplot
import numpy as np

N = 50
degree=2

x = np.random.rand(N,2)
y = x[:,0]*x[:,1] + 0.02*np.random.randn(N)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

m = np.arange(0, 1.05, 0.05)
X, Y = np.meshgrid(m,m)
x0 = np.array([np.ravel(X), np.ravel(Y)]).T
z0 = polyfit(x, y, x0, degree=degree)
Z = z0.reshape(X.shape)

ax.plot_wireframe(X, Y, Z, rcount=10, ccount=10, color='green')
ax.plot3D(x[:,0], x[:,1], y, 'o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
