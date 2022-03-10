from localreg import RBFnet
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,1,100)
y = np.zeros((len(x), 2))
y[:,0] = np.sin(2*np.pi*x)
y[:,1] = np.cos(2*np.pi*x)

net = RBFnet()
net.train(x, y)
yhat = net.predict(x)

plt.plot(x, y[:,0], 'C0', label='Ground truth')
plt.plot(x, y[:,1], 'C1', label='Ground truth')
plt.plot(x, yhat[:,0], ':k', label='Prediction')
plt.plot(x, yhat[:,1], ':k', label='Prediction')
plt.legend()
plt.show()
