"""
Optimizer for the NeuralNetwork model
"""

from GustavNet.nn import Model

class Optimizer:
    def step(self, net: Model) -> None:
        raise NotImplementedError

class SGD(Optimizer):

    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate
    
    def step(self, net: Model) -> None:
        for weights, grad in net.get_weights_and_grads():
            weights -= self.learning_rate * grad
        