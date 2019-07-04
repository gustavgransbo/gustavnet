"""
A NeuralNet that combines multiple different layers
"""

from GustavNet.tensor import Tensor
from GustavNet.layer import Layer
from typing import Sequence

class Model:

    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        
        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    