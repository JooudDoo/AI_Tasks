import numpy as np

from MST import BasicModule
from MST import MDT_REFACTOR_ARRAY

class Relu(BasicModule):
    def __init__(self):
        super().__init__()
    
    def forward(self, x) -> MDT_REFACTOR_ARRAY:
        self._inX = x
        self._outX = np.maximum(0, self._inX)
        return self._outX
    
    def backward_impl(self, dOut = None):
        """
            Производная по релу является занулением, значений производной по тем кооридантам, где у нас произошло зануления на `forward`
        """
        # !TODO переписать на более красиво
        _doutX = dOut.copy()
        _doutX[self._outX == 0] = 0
        return _doutX