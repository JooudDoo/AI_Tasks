import numpy as np

class MDT_REFACTOR_ARRAY(np.ndarray):
    #!TODO придумать нормальное название
    _source = None
    
    def backward(self, dOut = None):
        self._source._auto_backward(dOut)

    def copy(self, *args, **kwargs):
        cp = super().copy()
        cp._source = self._source
        return cp

def MDT_ARRAY(*args, **kwargs) -> MDT_REFACTOR_ARRAY:
    #!TODO придумать нормальное название
    a = np.array(*args, **kwargs)
    a.astype(np.float32)
    a = a.view(MDT_REFACTOR_ARRAY)
    a._source = None
    return a