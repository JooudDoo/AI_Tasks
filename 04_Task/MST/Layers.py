import numpy as np

from .BasicModules import BasicModule
from .MDT import MDT_REFACTOR_ARRAY, MDT_ARRAY

from .Addition.Functions import default_random_normal_dist__, ReLU_weight_init__

class FullyConnectedLayer(BasicModule):

    def __init__(self, in_size : int, out_size : int, use_bias = False):
        super().__init__()

        self._inX = None
        self._use_bias = use_bias
        self._w = ReLU_weight_init__(size=(in_size, out_size))
        self._bias = ReLU_weight_init__(size=(1, out_size)) if use_bias else np.zeros(shape=(1, out_size))

    def forward(self, x) -> MDT_REFACTOR_ARRAY:
        self._inX = x # .shape() = [batch_size, in_size]

        """
            Делаем сначало умножение матрицы на матрицу
            В итоге получаем матрицу [batch_size, out_size]
            С помощью np.tile делаем "broadcasting" для того чтобы учесть любый входной batchsize
        """
        self._outX = np.dot(x, self._w) + np.tile(self._bias, (x.shape[0], 1))
        return self._outX # .shape() = [batch_size, out_size]

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


class Conv2d(BasicModule):

    def __init__(self, inC : int, outC : int, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, use_bias = True, padding_value : float = 0):
        super().__init__()
        self._inC = inC
        self._outC = outC

        self._kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        self._padding = padding if isinstance(padding, tuple) else (padding, padding)
        self._padding = (self._padding[0] * 2, self._padding[1] * 2)
        self._padding_value = padding_value

        self._stride = stride if isinstance(stride, tuple) else (stride, stride)
        
        self._groups = groups # Not implemented 
        self._dilation = dilation

        self._use_bias = use_bias

        self.__init_weights()

    def __init_weights(self):
        self._w = ReLU_weight_init__(size=(self._outC, self._inC // self._groups, *self._kernel_size))
        self._bias = ReLU_weight_init__(size=(1, self._outC, 1, 1)) if self._use_bias else np.zeros(shape=(1, self._outC, 1, 1))

    def _calculate_output_sizes(self, inH : int, inW : int):
        """
            Вычисляет итоговый размер по обоим измерениям
        """
        outH = self.__calculate_output_dim_size(inH, 0)
        outW = self.__calculate_output_dim_size(inW, 1)
        return outH, outW
    
    def __calculate_output_dim_size(self, inSize : int, dim : int):
        """
            Вычисляет итоговый размер после свертки по измерению 
        """
        outSize = inSize
        outSize += self._padding[dim] # Add padding addition size
        outSize -= self._dilation*(self._kernel_size[dim]-1)
        outSize //= self._stride[dim]
        return outSize + 1

    def forward(self, x):
        BS, C, H, W = x.shape

        self.__outH, self.__outW = self._calculate_output_sizes(H, W) # Вычисляем размер изо после свертки

        self._inX = np.pad(x, [(0,0), (0,0), self._padding, self._padding], mode='constant', constant_values=self._padding_value) # Добавляем к входному изо паддинг

        self._outX = MDT_ARRAY(np.zeros((BS, self._outC, self.__outH, self.__outW))) # Создаем выходное изо

        for i in range(self.__outH):
            for j in range(self.__outW):
                input_patch = self._inX[:, :, i * self._stride[0]:i * self._stride[0] + self._kernel_size[0] * self._dilation:self._dilation,
                            j * self._stride[1]:j * self._stride[1] + self._kernel_size[1] * self._dilation:self._dilation]

                for k in range(self._groups):
                    input_group = input_patch[:, k * (self._inC // self._groups):(k + 1) * (self._inC // self._groups)]
                    weight_group = self._w[k * (self._outC // self._groups):(k + 1) * (self._outC // self._groups)]

                    self._outX[:, k * (self._outC // self._groups):(k + 1) * (self._outC // self._groups), i, j] = np.sum(
                        input_group[:, np.newaxis] * weight_group, axis=(2, 3, 4))


        self._outX += np.tile(self._bias, (BS, 1, self.__outH, self.__outW))

        return self._outX

    def forward_old(self, x):
        BS, C, H, W = x.shape

        self.__outH, self.__outW = self._calculate_output_sizes(H, W) # Вычисляем размер изо после свертки

        self._inX = np.pad(x, [(0,0), (0,0), self._padding, self._padding], mode='constant', constant_values=self._padding_value) # Добавляем к входному изо паддинг

        # Да благославит нас бог матана
        self._outX = np.zeros((BS, self._outC, self.__outH, self.__outW))
        # Вроде работает (Больше векторизации богу векторизации) !TODO
        for i in range(self.__outH):
            for j in range(self.__outW):
                # extract the region of the input
                x_part = self._inX[:, :, i*self._stride:i*self._stride+self._kernel_size[0], j*self._stride:j*self._stride+self._kernel_size[1]]
                # reshape the input and the filter for matrix multiplication
                x_part = x_part.reshape(BS, -1)
                w_reshape = self._w.reshape(self._outC, -1)
                # matrix multiplication and reshape back to output shape
                out_part = np.dot(x_part, w_reshape.T).reshape(BS, self._outC)
        
        self._outX += np.tile(self._bias, (BS, 1, self.__outH, self.__outW))
        
        return self._outX

    def backward_impl(self, dOut=None):
        ## REDO THIS !TODO
        BS, _, _, _ = dOut.shape

        self._dw = np.zeros_like(self._w)

        if self._use_bias:
            self._dbias = np.zeros_like(self._bias)

        self._dinX_pad = np.zeros_like(self._inX)
        
        for g in range(self._groups):
                x_part = self._inX[:, g*self._inC:(g+1)*self._inC, :, :]
                dout_part = dOut[:, g, :, :][:, np.newaxis, :, :, np.newaxis]
                self._dw[g] += np.sum(x_part[:, :, np.newaxis, :, :, :] * dout_part[:, np.newaxis, :, :, :, :], axis=0)
                dout_reshaped = np.broadcast_to(dout_part, (BS, self._outC // self._groups, self.__outH, self.__outW, self._inC, self._kernel_size[0], self._kernel_size[1]))
                self._dinX_pad[:, g*self._inC:(g+1)*self._inC, :, :] += np.sum(self._w[g][np.newaxis, :, :, :, :, :] * dout_reshaped[:, :, :, :, :, ::-1, ::-1], axis=(1, 4, 5, 6))
        

        # for i in range(self.__outH):
        #     for j in range(self.__outW):
        #         input_mask = self._inX[:, :, i*self._stride:i*self._stride+self._dilation*self._kernel_size[0]:self._dilation, j*self._stride:j*self._stride+self._dilation*self._kernel_size[1]:self._dilation]
        #         for k in range(self._outC):
        #             # Вычисляем градиенты для dX_pad, dW, и db
        #             self._dinX_pad[:, :, i*self._stride:i*self._stride+self._dilation*self._kernel_size[0]:self._dilation, j*self._stride:j*self._stride+self._dilation*self._kernel_size[1]:self._dilation] += self._w[k,:,:,:] * dOut[:,k,i,j][:,None,None,None]
        #             self._dw[k,:,:,:] += np.sum(input_mask * (dOut[:,k,i,j])[:,None,None,None], axis=0)
        
        if self._use_bias:
            self._dbias = np.sum(dOut, axis=(0, 2, 3), keepdims=True)
            
        # Удаление padding
        self._dinX = self._dinX_pad[:, :, self._padding[0]:-self._padding[0], self._padding[1]:-self._padding[1]]

        return self._dinX