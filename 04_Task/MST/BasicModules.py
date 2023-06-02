from numpy import ndarray
from .MDT import MDT_REFACTOR_ARRAY, MDT_ARRAY

class BasicModule:

    def __init__(self):
        self._modules : dict = {}
        self._parameters : list = []

        self.__zero_auto_backward_state()

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
    
    def __zero_auto_backward_state(self):
        self._hid_inX = None
        self._hid_outX = None
        self.__hidden_dOut = None
        self._hidden_links_count = 0 # the number of blocks that received our output value

    def _auto_backward_state(self):
        return self._hid_inX is not None

    def _add_hidden_output_link_count(self):
        self._hidden_links_count += 1

    def _add_hidden_out(self, dOut):
        if self.__hidden_dOut is None:
            self.__hidden_dOut = dOut
        else: # TODO check sizes
            self.__hidden_dOut += dOut

    def __call__(self, *args, **kwds):
        """
            Обработчик прямого прохода
        """
        self.__zero_auto_backward_state()
        for arg in args:
            if isinstance(arg, MDT_REFACTOR_ARRAY):
                if self._hid_inX is None:
                    self._hid_inX = [arg]
                else:
                    self._hid_inX.append(arg)
                if arg._source is not None: # If the source of our data is not None, then add to it the number of linked data references
                    arg._source._add_hidden_output_link_count()
        
        self._hid_outX = self.forward(*args, **kwds)
        if(not isinstance(self._hid_outX, MDT_REFACTOR_ARRAY) and isinstance(self._hid_outX, ndarray)):
            self._hid_outX = MDT_ARRAY(self._hid_outX)
        if self._hid_outX._source is None: # Если данные уже с меткой -> текущий модуль служебный и не имеет backward_impl
            self._hid_outX._source = self
        return self._hid_outX

    def forward(self):
        raise NotImplementedError(f"[{type(self).__name__}] is missing the required \"forward\" function")

    def backward(self, dOut = None):
        if self.__hidden_dOut is not None:
            if dOut is None:
                dOut = self.__hidden_dOut
            else:
                dOut += self.__hidden_dOut
        if self.backward_impl is not None:
            return self.backward_impl(dOut)
        raise NotImplementedError(f"[{type(self).__name__}] is missing the required \"backward_impl\" function")
    
    backward_impl = None

    def _auto_backward(self, dOut = None):
        """
            Если у нас есть `скрытые` выходные параметры => был произведен прямой проход => можно делать обратный

            Тогда проверяем есть ли у нашего модуля функция backprop
                - Если есть, обновляем градиенты с помощью нее
                - Если нет, ничего не делаем с градиентом

            Отправляем градиент всем модулям, которые являлись источниками данных для входа

            В конце зануляем состояния для проходов
        """
        self._hidden_links_count -= 1
        # If there are output values, you can backward (because forward was) Also check that all links were closed or this layer is the initiator of backward calculation
        if self._hid_outX is not None and (self._hidden_links_count == 0 or dOut is None):
            if self.backward_impl is not None:
                dOut = self.backward(dOut)
            if self._hid_inX is not None:
                for inArg in self._hid_inX:
                    if inArg._source is not None:
                        inArg._source._add_hidden_out(dOut)
                for inArg in self._hid_inX:
                    if inArg._source is not None:
                        inArg._source._auto_backward()

            self.__zero_auto_backward_state()

    @property
    def isTrainable(self):
        return self._w is not None or self._bias is not None
    
    
    def getModuleInfo(self, stringify = True):
        params = {"Trainable": self.isTrainable}

        if not stringify:
            return tuple(params.values())
        result_string = ""
        for param_name, param_value in params.items():
            if param_value is not None:
                result_string += f"{param_name}({param_value}) "
        return result_string

    def __stringify(self, module, result_string = None, depth=1):
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
            result_string += '\t' * (depth-1) + ' └── '

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
                result_string += module.getModuleInfo()
            else: # Если обьект не модуль, вызываем от него функцию рекурсивно
                result_string += f"\n{self.__stringify(module, depth=depth+1)}"
            result_string += "\n"

        return result_string

    def __str__(self):
        result_string = f"{self.__class__.__name__}:\n"
        return result_string + self.__stringify(self)

class Flatten(BasicModule):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        self.__inshape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward_impl(self, dOut = None):
        return dOut.reshape(self.__inshape)

class Sequential(BasicModule):
    """
        Класс для создания блоков с линейным проходом по всем его внутренним блокам
    """

    def __init__(self, *args : list[BasicModule]):
        super().__init__()
        for id, module in enumerate(args):
            self.set_by_name(f"__seq_layer_{id}", module)

    def forward(self, x):
        for module in self.get_modules().values():
            x = module(x)
        return x