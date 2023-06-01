import numpy as np

from MST import BasicModule
from MST import MDT_REFACTOR_ARRAY, MDT_ARRAY

from MST.Addition import HE_weight_init__

class Linear(BasicModule):

    def __init__(self, in_size : int, out_size : int, use_bias = False):
        super().__init__()

        self._inX = None
        self._use_bias = use_bias
        self._w = HE_weight_init__(size=(in_size, out_size))
        self._bias = np.zeros(shape=(1, out_size), dtype=np.float32)

    def forward(self, x) -> MDT_REFACTOR_ARRAY:
        self._inX = x
        self._outX = np.dot(x, self._w) + np.tile(self._bias, (x.shape[0], 1))
        return self._outX

    def backward_impl(self, dOut = None):        
        """
            Берем производную весов, по производной выхода. Т.е. dOut/dW
            inX.T.shape = (in_size, batch_size)
            dOut.shape  = (batch_size, out_size)
                    ------------Some Math--------------

            np.dot(self.inX, self._w) - операция прямого прохода
            Т.е. это матричное умножение, двух матриц разного размера
            Поэтому оно будет по сути является:

                l, m, n = batch_size, in_size, out_size
                inX.shape = (batch_size, in_size)
                _w.shape  = (in_size, out_size)

                sum_[0:m](inX[l_i] * _w[n_j])
                    Где: 
                        i = (0, 1, 2, ... l)
                        j = (0, 1, 2, ... n)

                Тогда производная от всего этого будет:
                    d( inX[l_i] * _w[n_j] ) = inX[l_i] * d(_w[n_j]) + _w[n_j] * d(inX[l_i])
                    А производная суммы, есть сумма производных:
                        d( inX[l_i] * _w[n_j] ) = inX[l_i] (если мы берем производную по _w)
                        d (sum_[0:m]{ inX[l_i] * _w[n_j] })/d( _w ) = d( np.dot() ) / d( _w ) ~~ inX 
                    Сл-но:
                        _w * inX = np.dot()
                        d(f) = d( np.dot() ) * inX.T

                        df/dw = df/da * da/d_w
                        df/da = известен, он есть dOut

                        da/d_w = d( np.dot() )/d( _w )
                        da/d_w = inX

                        Но т.к. (df/da).shape = (batch_size, out_size)
                        А inX.shape = (batch_size, in_size)
                        Мы берем inX.T с shape = (in_size, batch_size)

                        И в итоге получаем что наша производная равна

                        d_w = inX.T * dOut
                        Гдe:
                            dOut = df/da
        """
        self._dw = np.dot(self._inX.T, dOut)
        """
            Т.к. производная суммы есть сумма производных
            То тогда производная вокруг bias будет просто суммой значений производных df/da 
            (где a - текущий шаг, а f - то для чего мы считаем производную [затравка для начала расчета производной (loss - не совсем корректно)])
        """
        if self._use_bias:
            self._dbias = np.sum(dOut, axis=0, keepdims=True)

        """
            Производная по входным будет считаться аналогично производной для весов
            Т.к. между ними происходит операция матричного умножения
        """
        self._dinX = np.dot(dOut, self._w.T)
        return self._dinX