
import numpy as np

from .BasicModules import BasicModule

from .Addition.Functions import sigmoid

class SoftMax(BasicModule):
    #!TODO
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self._inX = x

        """
            Вычисляем максимум по батчу для стабильности вычисления
        """
        x_batch_max = np.max(x, axis=1, keepdims=True)
        x_exp = np.exp(x - x_batch_max)
        self._outX = x_exp / np.sum(x_exp, axis=1, keepdims=True)
        return self._outX

    def backward(self, dOut=None):
        if dOut is None:
            return np.ones_like(self._inX)
        raise NotImplementedError("Need to create backward")

class Sigmoid(BasicModule):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        self._inX = x

        self._outX = sigmoid(x)
        return self._outX

    def backward(self, dOut=None):
        # df/dinX = dOut * dsig/din
        return dOut * (sigmoid(self._outX)*(1 - sigmoid(self._outX)))

class Relu(BasicModule):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        self._inX = x
        self._outX = self._inX.copy()
        self._outX[self._outX < 0] = 0
        return self._outX
    
    def backward(self, dOut = None):
        if dOut is None:
            return np.ones_like(self._inX)
        
        """
            Производная по релу является занулением, значений производной по тем кооридантам, где у нас произошло зануления на `forward`
        """
        self._dinX = dOut.copy()
        self._dinX[self._inX < 0] = 0
        return self._dinX