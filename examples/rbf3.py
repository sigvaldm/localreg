from localreg import RBFnet
from localreg.metrics import rms_error, rms_rel_error
from frmt import print_table
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,0.49,100)
y = np.tan(np.pi*x)+1

net = RBFnet()

net.train(x, y, radius=1)
y_hat0 = net.predict(x)

net.train(x, y, radius=1, relative=True)
y_hat1 = net.predict(x)

print_table(
    [[''            , 'RMSE'              , 'RMSRE'                  ],
     ['Normal LLS'  , rms_error(y, y_hat0), rms_rel_error(y , y_hat0)],
     ['Relative LLS', rms_error(y, y_hat1), rms_rel_error(y , y_hat1)]]
)

plt.figure()
plt.plot(x, y, label='Ground truth')
plt.plot(x, y_hat0, label='Normal LLS')
plt.plot(x, y_hat1, label='Relative LLS')
plt.legend()
plt.show()
