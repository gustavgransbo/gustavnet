"""
Loss functions
"""
from GustavNet.tensor import Tensor
import numpy as np

class Loss:

    def loss(self, prediction: Tensor, target: Tensor) -> float:
        raise NotImplementedError
    
    def gradient(self, prediction: Tensor, target: Tensor) -> float:
        raise NotImplementedError

class MSE:
    """
    Mean squared error loss:
    J = (y - y_hat)^2
    """
    def loss(self, prediction: Tensor, target: Tensor) -> float:
        return np.mean(((prediction - target)**2))

    def gradient(self, prediction: Tensor, target: Tensor) -> Tensor:
        return 2 * (prediction - target)