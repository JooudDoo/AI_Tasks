
import numpy as np

from .BasicModules import BasicModule

class SGD:
    #!TODO with momentum

    _target_attributes = {
        '_w': '_dw',
        '_bias': '_dbias',
    }

    def __init__(self, module : BasicModule , lr : float, momentum : float = 0):
        self.modules : list[BasicModule] = self.__generate_modules_list(module)
        self.lr = lr
        self.momentum = momentum
        self.__create_modules_momentum_params()

    def __create_modules_momentum_params(self):
        for id, module in enumerate(self.modules):
            if self.momentum == 0:
                self.modules[id] = (None, module)
            else:
                velocity_params = {}
                for attribute, _ in self._target_attributes.items():
                    if module.get_by_name(attribute) is not None:
                        velocity_params.update({attribute: np.zeros_like(module.get_by_name(attribute))})
                self.modules[id] = (velocity_params, module)

    def __generate_modules_list(self, module, modules_d : list[BasicModule] = None):
        """
            Вытягиваем все обучаемые под модули модуля
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
                if sub_module.isTrainable():
                    modules_d.append(sub_module)
        return modules_d

    def _apply_optimization(self, module : BasicModule, attribute : str, params : dict[str, np.ndarray]): # sgd formula
        """
            v_(t+1) = momentum * v_(t) + grad { w_(t) }
            w_(t+1) = w_(t) - lr *v_(t+1)
        """
        if params is not None:
            params[attribute] = self.momentum * params[attribute] + module.get_by_name(self._target_attributes[attribute])
            velocity = params[attribute]
        else:
            velocity = module.get_by_name(self._target_attributes[attribute]) # Если моментум не используется то velocity = grad
        return module.get_by_name(attribute) - self.lr * velocity

    def step(self):
        for (params, module) in self.modules:
            bad_params = 0 # if all params of layer dont have any derivations
            for attribute, derivation in self._target_attributes.items():
                if module.get_by_name(derivation) is not None:
                    new_w = self._apply_optimization(module, attribute, params)
                    module.set_by_name(attribute, new_w)
                else:
                    bad_params += 1
                    if bad_params == len(self._target_attributes.items()):
                        if module._auto_backward_state():
                            raise RuntimeError(f"It is necessary to backward first before optimization")
                            
