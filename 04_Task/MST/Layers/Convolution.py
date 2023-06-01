
import numpy as np

from MST import BasicModule
from MST import MDT_REFACTOR_ARRAY, MDT_ARRAY

from MST.Addition import Relu_weight_init__

class Conv2d(BasicModule):

    def __init__(self, inC : int, outC : int, kernel_size, stride = 1, padding = 0, dilation = 1, use_bias = True, padding_value : float = 0):
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
        self._w = Relu_weight_init__(size=(self._outC, self._inC, *self._kernel_size))
        self._bias = np.zeros(shape=(1, self._outC, 1, 1), dtype=np.float32) #Relu_weight_init__(size=(1, self._outC, 1, 1)) if self._use_bias else 
    
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

        self._inX_cols = im2col_3(self._inX, self._output_size, self._kernel_size, self._stride) # .shape() = [C * K * K, BS * H * W]
        self._flatten_w = self._w.reshape(self._outC, -1) # .shape() = [outC, inC * K * K]

        self._outX = np.dot(self._flatten_w, self._inX_cols).reshape(self._outC, *self._output_size, BS)
        self._outX = self._outX.transpose(3, 0, 1, 2)

        if self._use_bias:
            self._outX += np.tile(self._bias, (BS, 1, *self._output_size))
        return self._outX
    
    def backward_impl(self, dOut=None):
        flatten_dOut = dOut.reshape(self._outC, -1)

        self._dw = np.dot(flatten_dOut, self._inX_cols.T).reshape(self._w.shape)

        self._dinX = np.dot(self._flatten_w.T, flatten_dOut)
        self._dinX = col2im_2(self._dinX, self._inX.shape, self._output_size, self._kernel_size, self._stride, self._padding)

        if self._use_bias:
            self._dbias = np.sum(dOut, axis=(0, 2, 3), keepdims=True)

        return self._dinX

def im2col_3(image, output_size, kernel_size, stride):

    _, C, _, _ = image.shape

    stride_h, stride_w = stride
    kernel_h, kernel_w = kernel_size

    output_height, output_width = output_size

    i0 = np.repeat(np.arange(kernel_h), kernel_w)
    i0 = np.tile(i0, C)
    i1 = stride_h * np.repeat(np.arange(output_height), output_width)
    j0 = np.tile(np.arange(kernel_w), kernel_h * C)
    j1 = stride_w * np.tile(np.arange(output_width), output_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), kernel_h * kernel_w).reshape(-1, 1)

    cols = image[:, k, i, j].transpose(1, 2, 0).reshape(kernel_h*kernel_w*C, -1)

    return cols

def im2col_2(image, output_size, kernel_size, stride):
    BS, C, _, _ = image.shape
    stride_h, stride_w = stride
    kernel_h, kernel_w = kernel_size

    output_height, output_width = output_size

    col = np.zeros((BS, output_height * output_width, C * kernel_h * kernel_w))

    for b in range(BS):
        for i in range(output_height):
            for j in range(output_width):
                patch = image[b, :, i * stride_h:i * stride_h + kernel_h, j * stride_w:j * stride_w + kernel_w]
                col[b, i * output_width + j, :] = np.reshape(patch, -1)

    return col

def im2col(image, output_size, kernel_size, stride):
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

def col2im_2(cols, image_shape, output_size, kernel_size, stride, padding):
    BS, C, H, W = image_shape
    stride_h, stride_w = stride
    kernel_h, kernel_w = kernel_size
    padding_h, padding_w = padding

    output_height, output_width = output_size

    cols = cols.reshape(kernel_h * kernel_w * C, -1, BS)
    cols = cols.transpose(2, 0, 1)

    image = np.zeros((BS, C, H, W))

    i0 = np.repeat(np.arange(kernel_h), kernel_w)
    i0 = np.tile(i0, C)
    i1 = stride_h * np.repeat(np.arange(output_height), output_width)
    j0 = np.tile(np.arange(kernel_w), kernel_h * C)
    j1 = stride_w * np.tile(np.arange(output_width), output_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), kernel_h * kernel_w).reshape(-1, 1)

    np.add.at(image, (slice(None), k, i, j), cols)

    if padding_h > 0:
        image = image[:, :, padding_h:-padding_h, :]
    if padding_w > 0:
        image = image[:, :, :, padding_w:-padding_w]
    
    return image

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

