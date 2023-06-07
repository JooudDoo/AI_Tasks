import numpy as np

from .BasicModules import BasicModule
from .MDT import MDT_REFACTOR_ARRAY

from .Activations import SoftMax

import torch.nn.functional as F
import torch

class CrossEntropyLoss(BasicModule):
    """
        Вычисляет CE loss для входных значений по origins

        Предварительно применяется softmax к входным данным

        Также при необходимости переводит origin к one_hot encoded вектору
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, origin) -> MDT_REFACTOR_ARRAY:
        self._inX = x.copy()
        self._batch_size = x.shape[0]
        self._origin = origin
        if x.shape != origin.shape:
            """
                Если получаем на вход разные размеры у x и origin
                То пытаемся преобразовать origin в one_hot вектора для каждого элемента батча
            """
            self._origin = np.zeros_like(x)
            self._origin[np.arange(self._batch_size), origin] = 1

        self._outX = -np.sum(np.log(self._inX) * self._origin) / self._batch_size
        return self._outX

    def backward_impl(self, dOut = None):
        if dOut is None:
            self._dinX = (self._inX - self._origin) / self._batch_size
            return self._dinX
        else:
            raise ValueError("TODO text for error")
        
class MAE(BasicModule):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, origin) -> MDT_REFACTOR_ARRAY:
        self._inX = x
        self._batch_size = x.shape[0]
        self._origin = origin

        return np.mean(np.abs(self._inX - self._origin))

    def backward_impl(self, dOut = None):
        if dOut is None:
            self._dinX = np.sign(self._origin - self._inX) / self._batch_size
            return self._dinX
        else:
            raise ValueError("TODO text for error")

class MSE(BasicModule):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, origin) -> MDT_REFACTOR_ARRAY:
        self._inX = x
        self._batch_size = x.shape[0]
        self._origin = origin

        return np.mean(np.power(self._inX - self._origin, 2))

    def backward_impl(self, dOut = None):
        if dOut is None:
            self._dinX = 2 * (self._origin - self._inX) / self._batch_size
            return self._dinX
        else:
            raise ValueError("TODO text for error")