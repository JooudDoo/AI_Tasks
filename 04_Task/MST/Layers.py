import numpy as np
from scipy import signal

from .BasicModules import BasicModule
from .MDT import MDT_REFACTOR_ARRAY, MDT_ARRAY

from .Addition import ReLU_weight_init__

class FullyConnectedLayer(BasicModule):

    def __init__(self, in_size : int, out_size : int, use_bias = False):
        super().__init__()

        self._inX = None
        self._use_bias = use_bias
        self._w = ReLU_weight_init__(size=(in_size, out_size))
        self._bias = ReLU_weight_init__(size=(1, out_size)) if use_bias else np.zeros(shape=(1, out_size))

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


class Conv2d(BasicModule):

    def __init__(self, inC : int, outC : int, kernel_size, stride = 1, padding = 0, dilation = 1, use_bias = False, padding_value : float = 0):
        super().__init__()
        self._inC = inC
        self._outC = outC

        self._kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        self._padding = padding if isinstance(padding, tuple) else (padding, padding)
        self._padding_value = padding_value

        self._stride = stride if isinstance(stride, tuple) else (stride, stride)
        
        self._dilation = dilation

        self._use_bias = use_bias

        self.__init_weights()

    def __init_weights(self):
        self._w = ReLU_weight_init__(size=(self._outC, self._inC, *self._kernel_size))
        self._bias = np.zeros(shape=(1, self._outC, 1, 1)) #ReLU_weight_init__(size=(1, self._outC, 1, 1)) if self._use_bias else 
    
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
        outSize += 2*self._padding[dim] # Add padding addition size
        outSize -= self._dilation*(self._kernel_size[dim]-1)+1
        outSize //= self._stride[dim]
        return outSize + 1
    
    def forward(self, x):
        BS, C, H, W = x.shape
        self._output_size = self._calculate_output_sizes(H, W)

        self._inX = np.pad(x, [(0,0), (0,0), self._padding, self._padding], mode='constant', constant_values=self._padding_value)

        self._inX_cols = img2col(self._inX, self._output_size, self._kernel_size, self._stride)
        self._flatten_w = self._w.reshape(self._outC, -1)

        self._outX = np.dot(self._flatten_w, self._inX_cols).reshape(self._outC, *self._output_size, BS)
        self._outX = self._outX.transpose(3, 0, 1, 2)

        if self._use_bias:
            self._outX += np.tile(self._bias, (BS, 1, *self._output_size))
        return self._outX
    
    def backward_impl(self, dOut=None):
        flatten_dOut = dOut.transpose(1, 2, 3, 0).reshape(self._outC, -1)

        self._dw = np.dot(flatten_dOut, self._inX_cols.T).reshape(self._w.shape)

        self._dinX = np.dot(self._flatten_w.T, flatten_dOut)
        self._dinX = col2im(self._dinX, self._inX.shape, self._output_size, self._kernel_size, self._stride, self._padding)

        if self._use_bias:
            self._dbias = np.sum(dOut, axis=(0, 2, 3), keepdims=True)

        return self._dinX

def img2col(image, output_size, kernel_size, stride):
    BS, C, _, _ = image.shape
    stride_h, stride_w = stride
    kernel_h, kernel_w = kernel_size

    output_height, output_width = output_size

    patches = np.zeros((BS, C, kernel_h, kernel_w, output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            patches[:, :, :, :, i, j] = image[:, :, i * stride_h:i * stride_h + kernel_h,
                                          j * stride_w:j * stride_w + kernel_w]
            
    col_matrix = patches.transpose(0, 4, 5, 1, 2, 3).reshape(BS * output_height * output_width, -1).T

    return col_matrix

def col2im(cols, image_shape, output_size, kernel_size, stride, padding):
    BS, C, H, W = image_shape

    patches = cols.T.reshape(BS, *output_size, C, *kernel_size).transpose(0, 3, 4, 5, 1, 2)

    stride_h, stride_w = stride
    crop_h, crop_w = kernel_size
    padding_h, padding_w = padding

    image = np.zeros((BS, C, H, W))
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            image[:, :, i * stride_h:i * stride_h + crop_h, j * stride_w:j * stride_w + crop_w] += patches[:, :, :, :, i, j]

    if padding_h > 0:
        image = image[:, :, padding_h:-padding_h, :]
    if padding_w > 0:
        image = image[:, :, :, padding_w:-padding_w]

    return image

