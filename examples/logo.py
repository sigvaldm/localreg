import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D # Axes3D import has side effects, it enables using projection='3d' in add_subplot
import numpy as np

cdict = {'red':   [(0.0,  1.0, 1.0),
                   (1.0,  0.12, 0.12)],
         'green': [(0.0,  1.0, 1.0),
                   (1.0,  0.46, 0.46)],
         'blue':  [(0.0,  1.0, 1.0),
                   (1.0,  0.83, 0.83)],
         'alpha':  [(0.0,  0.0, 0.0),
                   (1.0,  1.0, 1.0)]}

cmap = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

points = np.array([[0,0],
                   [1.8,0],
                   [1.3,-1.5]])
amps = [1, 0.8, 0.7]

xmax = max(points[:,0])
xmin = min(points[:,0])
ymax = max(points[:,1])
ymin = min(points[:,1])

x = np.arange(xmin-3, xmax+3, 0.01)
y = np.arange(ymin-3, ymax+3, 0.01)
X, Y = np.meshgrid(x, y)

for p, a in zip(points, amps):
    Z = a*np.exp(-(X-p[0])**2-(Y-p[1])**2)
    # Z[np.where(Z<0.1*a)] = np.nan
    ax.plot_surface(X, Y, Z, cmap=cmap, rcount=300, ccount=300)

x = np.arange(xmin-1, xmax+1, 0.1)
y = np.arange(ymin-1, ymax+1, 0.1)
X, Y = np.meshgrid(x, y)

Z = np.zeros_like(X)
for p, a in zip(points, amps):
    Z += a*np.exp(-(X-p[0])**2-(Y-p[1])**2)
ax.plot_wireframe(X, Y, Z, color='darkgreen', alpha=0.3)

ax.set_xlim((xmin-1, xmax+1))
ax.set_ylim((ymin-1, ymax+1))

ax.axis('off')

plt.savefig('logo.png')
plt.show()
