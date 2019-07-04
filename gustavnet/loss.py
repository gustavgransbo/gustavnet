"""
Loss functions
"""
from gustavnet.tensor import Tensor
import numpy as np

class Loss:

    def loss(self, prediction: Tensor, target: Tensor) -> float:
        raise NotImplementedError
    
    def gradient(self, prediction: Tensor, target: Tensor) -> float:
        raise NotImplementedError

class MSE(Loss):
    """
    Mean squared error loss:
    J = (y - y_hat)^2
    """
    def loss(self, prediction: Tensor, target: Tensor) -> float:
        return np.mean(((prediction - target)**2))

    def gradient(self, prediction: Tensor, target: Tensor) -> Tensor:
        return 2 * (prediction - target)

class BinaryCrossEntropy(Loss):
    """
    Binary Cross Entropy Loss
    J = y * log(y_hat) + (1 - y)log(1 - y_hat)
    """
    def loss(self, prediction: Tensor, target: Tensor) -> float:
        return np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))

    def gradient(self, prediction: Tensor, target: Tensor) -> Tensor:
        # Inefficient to do do matrix inverse. 
        # Since Binary Cross Entropy is commonly used together with
        # the sigmoid activation function, implementing a combined
        # sigmoid + binary cross entropy loss would be a good idea
        # See: https://discuss.pytorch.org/t/bceloss-vs-bcewithlogitsloss/33586/8

        # Avoid dividing by 0
        prediction = np.clip(prediction, 1e-15, 1 - 1e-15)
        return (prediction - target) / (prediction - prediction**2)