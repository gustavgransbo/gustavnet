"""
A NeuralNet that combines multiple different layers
"""

from gustavnet.tensor import Tensor
from gustavnet.layer import Layer
from typing import Sequence, Tuple, Iterator

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

    def get_weights_and_grads(self) -> Iterator[Tuple[str, Tensor]]:
        for layer in self.layers:
            for name, weights in layer.params.items():
                yield weights, layer.gradients[name]

    