
from MST import BasicModule

class ConvBasicModule(BasicModule):
    
    def __init__(self, kernel_size, stride = 1, padding = 0, dilation = 1, padding_value : float = 0):
        super().__init__()

        self._kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        self._padding = padding if isinstance(padding, tuple) else (padding, padding)
        self._padding_value = padding_value

        self._stride = stride if isinstance(stride, tuple) else (stride, stride)
        
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