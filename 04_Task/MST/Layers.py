import numpy as np

from .BasicModules import BasicModule

from .Addition.Functions import default_random_normal_dist__, relu_weight_init__

class FullyConnectedLayer(BasicModule):

    def __init__(self, in_size : int, out_size : int, use_bias = False):
        super().__init__()

        self._inX = None
        self._use_bias = use_bias
        self._w = relu_weight_init__(size=(in_size, out_size))
        self._bias = relu_weight_init__(size=(1, out_size)) if use_bias else np.zeros(shape=(1, out_size))

    def forward(self, x):
        self._inX = x # .shape() = [batch_size, in_size]

        """
            Делаем сначало умножение матрицы на матрицу
            В итоге получаем матрицу [batch_size, out_size]
            С помощью np.tile делаем "broadcasting" для того чтобы учесть любый входной batchsize
        """
        self._outX = np.dot(x, self._w) + np.tile(self._bias, (x.shape[0], 1))
        return self._outX # .shape() = [batch_size, out_size]

    def backward(self, dOut = None):        
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


class Conv2d(BasicModule):

    def __init__(self, inC : int, outC : int, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, use_bias = True, padding_value : float = 0):
        super().__init__()
        self._inC = inC
        self._outC = outC

        self._kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        self._padding = padding if isinstance(padding, tuple) else (padding, padding)
        self._padding_value = padding_value

        self._stride = stride
        
        self._groups = groups
        self._dilation = dilation

        self._use_bias = use_bias

        self.__init_weights()

    def __init_weights(self):
        self._w = default_random_normal_dist__((self._outC, self._inC // self._groups, *self._kernel_size))
        self._b = default_random_normal_dist__(size=(1, self._outC, 1, 1)) if self._use_bias else np.zeros(shape=(1, self._outC, 1, 1))

    def _calculate_output_sizes(self, inH : int, inW : int):
        """
            Вычисляет итоговый размер по обоим измерениям
        """
        outH = self.__calculate_output_dim_size(inH, 0)
        outW = self.__calculate_output_dim_size(inW, 1)
        return outH, outW
    
    def __calculate_output_dim_size(self, inSize : int, sizeType : int):
        """
            Вычисляет итоговый размер после свертки по измерению 
        """
        outSize = inSize
        outSize += self._padding[sizeType] # Add padding addition size
        outSize -= self._dilation*(self._kernel_size[sizeType]-1)-1
        outSize //= self._stride
        return outSize

    def forward(self, x):
        BS, C, H, W = x.shape

        outH, outW = self._calculate_output_sizes(H, W) #Вычисляем размер изо после свертки

        self._inX = np.pad(x, self._padding, mode='constant', constant_values=self._padding_value) # Добавляем к входному изо паддинг

        convResult = np.zeros((BS, self._outC, outH, outW))
        for channel in range(self._inC):
            for v_shift in range(0, H - self._kernel_size[1], self._stride):
                for h_shift in range(0, W - self._kernel_size[0], self._stride):
                    convResult[:, channel, v_shift, h_shift] = np.sum(
                        self._inX[:, channel, v_shift:v_shift+self._kernel_size[1], h_shift:h_shift+self._kernel_size[0]] * self._w
                    )
        
        return convResult

    def backward(self, dOut=None):
        return self._dinX

if __name__ == '__main__':
    c = Conv2d(3, 9, 3, padding=1)

    