import numpy as np

def MDT_ARRAY(*args, **kwargs):
    #!TODO придумать нормальное название
    a = np.array(*args, **kwargs)
    a = a.view(MDT_REFACTOR_ARRAY)
    a._source = None
    return a

class MDT_REFACTOR_ARRAY(np.ndarray):
    #!TODO придумать нормальное название
    _source = None
    
    def backward(self, dOut = None):
        self._source.backward(dOut)