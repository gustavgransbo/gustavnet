"""
A layer has two functions:
1. It propagates input forward
2. It propagates a gradient backwards
"""

from GustavNet.tensor import Tensor
from typing import Dict
import numpy as np

class Layer:

    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.gradients: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError

class Dense(Layer):
    """
    This dense layer takes an input x and calculates output as y = xW + b
    Where the dimensions of all factors are:
    x: BxN
    W: NxM
    b: M
    y: M
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()

        # TODO: Don't use a normal distribution to initialize weights
        self.params['W'] = np.random.randn(input_size, output_size)
        self.params['b'] = np.random.randn(output_size)
    
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Propagates input (x) forward acording to:
        y = xW + b
        """
        # Save inputs for backpropagation
        self.inputs = inputs
        return  inputs @ self.params['W'] + self.params['b']

    def backward(self, grad: Tensor) -> Tensor:
        """
        Propagates a gradiant (dJ/dy) backwards through the layer

        Derivates for the fully connected layers parameters:
        dJ/dW = dy/dW * dJ/dy, dy/dW = x
        dJ/dx = dy/dx * dJ/dy, dy/dx = W
        dJ/db = dy/db * dJ/dy, dy/db = 1
        """
        self.gradients['W'] = self.inputs.T @ grad
        self.gradients['b'] = grad.sum(0)

        return grad @ self.params['W'].T




