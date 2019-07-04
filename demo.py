"""
Solves the XOR problem using gustavnet
"""

from gustavnet.nn import Model
from gustavnet.tensor import Tensor
from gustavnet.layer import Dense, Tanh, Sigmoid
from gustavnet.loss import MSE, BinaryCrossEntropy
from gustavnet.optim import SGD
import numpy as np

net = Model(
    [Dense(2, 2),
    Tanh(),
    Dense(2, 1),
    Sigmoid()]
 )

X = np.asarray([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
Y = np.asarray([
    [0],
    [1],
    [1],
    [0]
])
mse_loss = BinaryCrossEntropy()
sgd_optimizer = SGD(learning_rate=0.1)


for _ in range(10000):
    pred = net.forward(X)
    net.backward(mse_loss.gradient(pred, Y))
    sgd_optimizer.step(net)
    
    loss = mse_loss.loss(pred, Y)
    print("MSE: %.4f" % loss)
    
for x, y in zip(X, Y):
    pred = net.forward(x)
    print(x, pred.round(2), y)