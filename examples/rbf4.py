from localreg import RBFnet
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,1,100)
y = np.zeros((len(x), 2))
y[:,0] = np.sin(2*np.pi*x)
y[:,1] = np.cos(2*np.pi*x)

net = RBFnet()
net.train(x, y, num=10, radius=0.3)

yhat = net.predict(x)

plt.plot(x, y[:,0], label='Ground truth')
plt.plot(x, y[:,1], label='Ground truth')
plt.plot(x, yhat[:,0], label='Prediction')
plt.plot(x, yhat[:,1], label='Prediction')
plt.legend()
plt.show()
