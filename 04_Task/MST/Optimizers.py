
from .BasicModules import BasicModule

class SGD:
    #!TODO
    #!TODO with momentum

    _target_attributes = {
        '_w': '_dw',
        '_bias': '_dbias',
    }

    def __init__(self, module : BasicModule , lr : float):
        self.modules : list[BasicModule] = self.__generate_modules_list(module)
        self.lr = lr

    def __generate_modules_list(self, module, modules_d : list[BasicModule] = None):
        """
            Extracting all modules inside main module
            Return: list of all modules
        """
        if modules_d is None:
            modules_d = []
        
        if isinstance(module, BasicModule):
            _extr_place = module._extract_sub_modules().items()
        elif isinstance(module, dict):
            _extr_place = module.items()
        else:
            raise RuntimeError(f"It is not possible to pull parameters from {module}")

        for sub_moduleName, sub_module in _extr_place:
            if isinstance(sub_module, dict):
                modules_d = self.__generate_modules_list(sub_module, modules_d)
            else:
                modules_d.append(sub_module)
        return modules_d

    def _apply_optimization(self, w, dW): # sgd formula
        """
            w_(t+1) = w_(t) - lr * grad{ w_(t) }
        """
        return w - self.lr * dW

    def step(self):
        for module in self.modules:
            for attribute, derivation in self._target_attributes.items():
                if module.get_by_name(attribute) is not None: #Optimize weights
                    if module.get_by_name(derivation) is not None:
                        new_w = self._apply_optimization(module.get_by_name(attribute), module.get_by_name(derivation))
                        module.set_by_name(attribute, new_w)
                    else:
                        raise RuntimeError(f"It is necessary to backward first to perform optimization")
