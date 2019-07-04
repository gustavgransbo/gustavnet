"""
Solves the XOR problem using GustavNet
"""

from GustavNet.nn import Model
from GustavNet.tensor import Tensor
from GustavNet.layer import Dense, Tanh
from GustavNet.loss import MSE
from GustavNet.optim import SGD
import numpy as np

net = Model(
    [Dense(2, 2),
    Tanh(),
    Dense(2, 2)]
 )

X = np.asarray([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
Y = np.asarray([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])
mse_loss = MSE()
sgd_optimizer = SGD()


for _ in range(5000):
    pred = net.forward(X)
    net.backward(mse_loss.gradient(pred, Y))
    sgd_optimizer.step(net)
    
    loss = mse_loss.loss(pred, Y)
    print("MSE: %.4f" % loss)
    
for x, y in zip(X, Y):
    pred = net.forward(x)
    print(x, pred.round(2), y)

    