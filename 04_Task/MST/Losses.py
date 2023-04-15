import numpy as np

from .BasicModules import BasicModule

class CrossEntropyLoss(BasicModule):
    #!REDO
    def __init__(self):
        super().__init__()

    def forward(self, x, origin):
        self._batch_size = x.shape[0]

        if x.shape != origin.shape:
            """
                Если получаем на вход разные размеры у x и origin
                То пытаемся преобразовать origin в one_hot вектора для каждого элемента батча
            """
            self._origin = np.zeros_like(x)
            self._origin[np.arange(self._batch_size), origin] = 1
        self._inX = x
        
        self._outX = -np.sum(self._origin * np.log(x)) / self._batch_size
        return self._outX

    def backward(self, dOut = None):
        #!REDO
        #Разобраться
        if dOut is None:
            return (self._inX - self._origin) / self._batch_size
            # return np.ones_like(self._inX)
        else:
            raise ValueError("TODO text for error")