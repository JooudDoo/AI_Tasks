
from .BasicModule import BasicModule

class SGD:
    #!TODO
    #!TODO with momentum

    _target_attributes = {
        '_w': '_dw',
        '_bias': '_dbias',
    }

    def __init__(self, module : BasicModule , lr : float):
        self.modules : dict[str, BasicModule] = module.get_modules()
        self.lr = lr

    def _apply_optimization(self, w, dW): # sgd formula
        """
            w_(t+1) = w_(t) - lr * grad{ w_(t) }
        """
        return w - self.lr * dW

    def step(self):
        for moduleName, module in self.modules.items():
            for attribute, derivation in self._target_attributes.items():
                if module.get_by_name(attribute) is not None: #Optimize weights
                    if module.get_by_name(derivation) is not None:
                        new_w = self._apply_optimization(module.get_by_name(attribute), module.get_by_name(derivation))
                        module.set_by_name(attribute, new_w)
                    else:
                        raise RuntimeError(f"It is necessary to backward first to perform optimization")
