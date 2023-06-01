import numpy as np

from MST import BasicModule
from MST import MDT_REFACTOR_ARRAY

from MST.Addition import sigmoid

class Sigmoid(BasicModule):
    #!TODO вынести функции сигмоиды в папку Addition
    def __init__(self):
        super().__init__()
    
    def forward(self, x) -> MDT_REFACTOR_ARRAY:
        self._inX = np.clip(x, -600, 600)
        self._outX = sigmoid(self._inX)
        return self._outX

    def backward_impl(self, dOut = None):
        #!REDO разобраться с производной сигмоиды
        # df/dinX = dOut * dsig/din
        return dOut * (self._outX*(1 - self._outX))
