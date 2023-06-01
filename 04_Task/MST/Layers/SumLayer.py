
import numpy as np

from MST import BasicModule
from MST import MDT_REFACTOR_ARRAY, MDT_ARRAY

class Sum(BasicModule):

    def __init__(self):
        super().__init__()

    def forward(self, x_1, x_2):
        self._inX_1 = x_1
        self._inX_2 = x_2
        return x_1 + x_2

    def backward_impl(self, dOut=None):
        return dOut