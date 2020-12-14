from localreg import RBFnet
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,1,100)
y = np.sin(2*np.pi*x)

net = RBFnet()
net.train(x, y, num=10, radius=0.3)

plt.plot(x, y, label='Ground truth')
net.plot_bases(plt.gca(), x, label='Prediction')
plt.legend()
plt.show()
