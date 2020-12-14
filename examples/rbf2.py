from localreg import RBFnet, plot_corr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Enables 3d-projection 

x = np.linspace(0,2,30)
X, Y = np.meshgrid(x, x)

input = np.array([X.ravel(), Y.ravel()]).T
x, y = input.T
z = y*np.sin(2*np.pi*x)

net = RBFnet()
net.train(input, z, num=50)
z_hat = net.predict(input)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, z.reshape(X.shape), rcount=20, ccount=20)
ax.plot_surface(X, Y, z_hat.reshape(X.shape), alpha=0.5, color='green')
plt.show()

fig, ax = plt.subplots()
plot_corr(ax, z, z_hat)
plt.show()
