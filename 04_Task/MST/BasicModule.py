import numpy

class BasicModule:

    def __init__(self):
        self._modules : dict = {}
        self._parameters : list = []

        self._w = None
        self._bias = None
        self._dw = None
        self._dbias = None
    
    def __setattr__(self, __name: str, __value) -> None:
        if isinstance(__value, BasicModule):
            self._modules[__name] = __value
        super().__setattr__(__name, __value)

    def __delattr__(self, __name: str) -> None:
        if self._modules.get(__name, None) is not None:
            self._modules.pop(__name)
        super().__delattr__(__name)

    def get_by_name(self, name: str):
        return self.__getattribute__(name)
    
    def set_by_name(self, name: str, value):
        self.__setattr__(name, value)

    def get_modules(self):
        if len(self._modules) != 0:
            return self._modules
        raise RuntimeError(f"This module has no internal BasicModules")

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self):
        raise NotImplementedError(f"[{type(self).__name__}] is missing the required \"forward\" function")

    def backward(self, dOut = None):
        raise NotImplementedError(f"[{type(self).__name__}] is missing the required \"backward\" function")