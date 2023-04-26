
import numpy as np

from .BasicModules import BasicModule

from .Addition.Functions import sigmoid

class Sigmoid(BasicModule):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        self._inX = np.clip(x, -600, 600) # Используем clip для предотвращения переполнения экспоненты
        self._outX = sigmoid(self._inX)
        return self._outX

    def backward(self, dOut = None):
        #!REDO разобраться с производной сигмоиды
        # df/dinX = dOut * dsig/din
        return dOut * (self._outX*(1 - self._outX))

class ReLu(BasicModule):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        self._inX = x
        self._outX = np.maximum(0, self._inX)
        return self._outX
    
    def backward(self, dOut = None):
        """
            Производная по релу является занулением, значений производной по тем кооридантам, где у нас произошло зануления на `forward`
        """

        # !TODO переписать на более красиво
        _doutX = dOut.copy()
        _doutX[self._outX == 0] = 0
        return _doutX

class leakyReLu(BasicModule):
    def __init__(self, scale : float = 0.01):
        super().__init__()
        self.scale = scale
    
    def forward(self, x):
        self._inX = x
        self._outX = self._inX.copy()
        self._outX[self._inX < 0] *= self.scale
        return self._outX
    
    def backward(self, dOut = None):
        """
            
        """
        dOut[self._inX < 0] *= self.scale
        return dOut