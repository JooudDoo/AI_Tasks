
import numpy as np

from MST import BasicModule
from MST import MDT_REFACTOR_ARRAY
from ..Addition import softMax

class SoftMax(BasicModule):
    def __init__(self):
        super().__init__()

    def forward(self, x) -> MDT_REFACTOR_ARRAY:
        self._inX = x.copy()
        self._outX = softMax(x) + 1e-12
        return self._outX 
    
    def backward_impl(self, dOut=None):
        self._dinX = (dOut - (dOut * self._outX).sum(axis=1, keepdims=True)) * self._outX
        return self._dinX