
import numpy as np
import warnings

from MST import BasicModule

class ConvBasicModule(BasicModule):
    
    def __init__(self, kernel_size, stride = 1, padding = 0, dilation = 1, padding_value : float = 0):
        super().__init__()

        self._kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        self._padding = padding if isinstance(padding, tuple) else (padding, padding)
        self._padding_value = padding_value

        self._stride = stride if isinstance(stride, tuple) else (stride, stride)
        
        if dilation != 1:
            raise warnings.warn(f"This argument is not currently supported Dilation", RuntimeWarning)
        self._dilation = dilation

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
    
    def __im2col_indices_calc(self, image_shape):
        _, C, H, W = image_shape
        stride_h, stride_w = self._stride
        kernel_h, kernel_w = self._kernel_size

        output_height, output_width = self._output_size

        i0 = np.repeat(np.arange(kernel_h), kernel_w)
        i0 = np.tile(i0, C)
        i1 = stride_h * np.repeat(np.arange(output_height), output_width)
        j0 = np.tile(np.arange(kernel_w), kernel_h * C)
        j1 = stride_w * np.tile(np.arange(output_width), output_height)

        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(C), kernel_h * kernel_w).reshape(-1, 1)
        return (i, j, k)
    
    def  _im2col(self, image):
        _, C, H, W = image.shape

        kernel_h, kernel_w = self._kernel_size

        i, j, k =  self.__im2col_indices_calc(image.shape)

        cols = image[:, k, i, j].transpose(1, 2, 0).reshape(kernel_h*kernel_w*C, -1)

        return cols
    
    def _col2im(self, cols, image_shape):
        BS, C, H, W = image_shape

        padding_h, padding_w = self._padding
        kernel_h, kernel_w = self._kernel_size

        cols = cols.reshape(kernel_h * kernel_w * C, -1, BS)
        cols = cols.transpose(2, 0, 1)

        image = np.zeros((BS, C, H, W))

        i, j, k = self.__im2col_indices_calc(image_shape)

        np.add.at(image, (slice(None), k, i, j), cols)

        if padding_h > 0:
            image = image[:, :, padding_h:-padding_h, :]
        if padding_w > 0:
            image = image[:, :, :, padding_w:-padding_w]
        
        return image