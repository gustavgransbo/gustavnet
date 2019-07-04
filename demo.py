"""
A demo of the package
"""

from GustavNet.nn import Model
from GustavNet.tensor import Tensor
from GustavNet.layer import Dense
from GustavNet.loss import MSE
import numpy as np

net = Model([Dense(20, 10), Dense(10, 1)])

X = np.random.randn(2, 20)
y = np.ones((2,1))
loss = MSE()

pred = net.forward(X)
grad = net.backward(loss.gradient(pred, y))