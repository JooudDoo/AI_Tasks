import numpy as np

from .BasicModules import BasicModule
from .MDT import MDT_REFACTOR_ARRAY

from .Addition import softMax

class CrossEntropyLoss(BasicModule):
    """
        Вычисляет CE loss для входных значений по origins

        Предварительно применяется softmax к входным данным

        Также при необходимости переводит origin к one_hot encoded вектору
    """
    #!REDO
    # Скорее всего нужно делать другую инициализацию весов
    # Некорректная работа с функцией активации ReLU
    # Приходят слишком большие значение на вход или одни нули
    # В итоге после софтмакс получаем что градиенты назада тякут вяло
    # решение - ???
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __init__(self):
        super().__init__()

    def forward(self, x, origin) -> MDT_REFACTOR_ARRAY:
        self._inX = x
        self._batch_size = x.shape[0]
        self._origin = origin
        if x.shape != origin.shape:
            """
                Если получаем на вход разные размеры у x и origin
                То пытаемся преобразовать origin в one_hot вектора для каждого элемента батча
            """
            self._origin = np.zeros_like(x)
            self._origin[np.arange(self._batch_size), origin] = 1

        """
            Применяем softMax чтобы привести входы к вектору вероятностей
            Добавляем очень маленькое число для стабильности
        """
        self._smOut = softMax(x) + 1e-9

        self._outX = -np.sum(self._origin * np.log(self._smOut)) / self._batch_size
        return self._outX

    def backward_impl(self, dOut = None):
        """
            Вычисление градиента по CE в итоге будет разницей между softMax(inX) и y_gt
            Это можно получить из:

            forward:
                S - softMax далее
                S(x) = (e^t_i) / ( sum {e^t_i} ) 
                    Где: t_i <- self._inX
                
                Тогда:
                    CE(S(t), y) = -sum { y_i * ln( S(t_i) ) }
                        Где y <- self._origin
            
            backward:
                dE/dt = -y + S(t)

                Это получается из того что:
                    CE(S(t), y) = - sum{ y_i * ln(S(t_i)) } =
                        Выносим e^t_i из под логарифма:
                      = -sum_[i]{ y_i * (t_i - ln(sum_[j]{ e^t_j }) ) } =
                        Разбиваем сумму на под суммы
                      = -sum_[i]{ y_i * t_i } + sum_[i]{ y_i } * ln( sum_[j]{ e^t_j } ) =
                        Т.к. sum_[i]{ y_i } = 1 (y - one hot encoded vector)
                      =  -sum_[i]{ y_i * t_i } + ln( sum_[j]{ e^t_j } ) 
                    Получаем в итоге зависимость между y_i и t_i
                
                Тогда dE/dt = 
                    = -sum[i]{ y_i } + ( 1 / {sum_[j]< e^t_j > } ) * e^t =
                        ( 1 / {sum_[j]< e^t_j > } ) * e^t ~~ SoftMax
                    = -sum[i]{ y_i } + S(t) =
                    = -y + S(t)
        """
        if dOut is None:
            dLoss = (self._smOut - self._origin) / self._batch_size
            if self._hid_inX is not None:
                for inX in self._hid_inX:
                    if inX._source is not None:
                        inX._source._auto_backward(dLoss)
        else:
            dLoss = (self._smOut - self._origin) / self._batch_size
            if self._hid_inX is not None:
                for inX in self._hid_inX:
                    if inX._source is not None:
                        inX._source._auto_backward(dLoss)
            # raise ValueError("TODO text for error")