
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

    def _extract_sub_modules(self, module = None, modules_d : dict = None) -> dict:
        """
            Функция возвращает словарь содержащий все под модули модуля
            Если внутри модуля есть Sequential он будет раскрыт и преобразован в dict
        """
        if module is None:
            module = self
        if module.get_modules is None:
            return None
        if modules_d is None:
            modules_d = {}
        for sub_moduleName, sub_module in module.get_modules().items():
            if sub_module.get_modules() is not None:
                modules_d.update({sub_moduleName: self._extract_sub_modules(sub_module)})
            else:
                modules_d.update({sub_moduleName:sub_module})
        return modules_d

    def get_modules(self):
        if len(self._modules) != 0:
            return self._modules
        return None

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self):
        raise NotImplementedError(f"[{type(self).__name__}] is missing the required \"forward\" function")

    def backward(self, dOut = None):
        raise NotImplementedError(f"[{type(self).__name__}] is missing the required \"backward\" function")
    
    def __stringify(self, module, result_string = None, depth=0):
        #!TODO get module information from module
        if result_string is None:
            result_string = ""

        if isinstance(module, BasicModule):
            _extract_plc = module._extract_sub_modules()
        elif isinstance(module, dict):
            _extract_plc = module
        else:
            raise RuntimeError(f"It is not possible to pull information from {module}")

        for moduleName, module in _extract_plc.items():
            result_string += '\t' * depth

            # Достаем имя модуля
            if "__seq_layer_" in moduleName:
                if  module.__class__.__name__ == 'dict':
                    result_string += "Sequential" + ": "
                else:
                    result_string += module.__class__.__name__ + ": "
            else:
                result_string += moduleName + ": "
            
            # Достаем информацию о модуле
            if isinstance(module, BasicModule):
                result_string += f"module_information"
            else: # Если обьект не модуль, вызываем от него функцию рекурсивно
                result_string += f"\n{self.__stringify(module, depth=depth+1)}"
            result_string += "\n"

        return result_string

    def __str__(self):
        result_string = f"{self.__class__.__name__}:\n"
        return result_string + self.__stringify(self, depth=1)


class Sequential(BasicModule):
    """
        Класс для создания блоков с линейным проходов по всем его внутренним блокам
    """

    def __init__(self, *args : list[BasicModule]):
        super().__init__()
        for id, module in enumerate(args):
            self.set_by_name(f"__seq_layer_{id}", module)

    def forward(self, x):
        for module in self.get_modules().values():
            x = module(x)
        return x
    
    def backward(self, dOut=None):
        dN = dOut
        for module in list(self.get_modules().values())[::-1]:
            dN = module.backward(dN)
        return dN