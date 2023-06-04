
import numpy as np

from MST import MDT_REFACTOR_ARRAY, MDT_ARRAY

from .Basic import ConvBasicModule

class MaxPool2d(ConvBasicModule):
    def __init__(self, kernel_size, stride = None, padding = 0, dilation = 1, padding_value = 0):
        if stride is None:
            stride = kernel_size
        super().__init__(kernel_size, stride, padding, dilation, padding_value)

    def forward(self, x):
        BS, C, H, W = x.shape
        self._output_size = self._calculate_output_sizes(H, W)
        self._inX = x
        self._inX_padded = np.pad(x, [(0,0), (0,0), (self._padding[0], self._padding[0]), (self._padding[1], self._padding[1])], mode='constant', constant_values=self._padding_value)
        self._inX_cols = self._inX_padded.reshape(BS * C, 1, H+self._padding[0]*2, W+self._padding[1]*2)
        self._inX_cols = self._im2col(self._inX_cols)  # .shape() = [C, K * K, BS * outH * outW]

        self._max_idx = np.argmax(self._inX_cols, axis=0) 

        self._outX = self._inX_cols[self._max_idx, range(self._max_idx.size)]

        self._outX = self._outX.reshape(*self._output_size, BS, C)
        self._outX = self._outX.transpose(2, 3, 0, 1)

        return self._outX
    
    def backward_impl(self, dOut=None):
        flatten_dOut = dOut.transpose(2, 3, 0, 1).ravel()

        self._dinX = np.zeros_like(self._inX_cols)
        self._dinX[self._max_idx, range(self._max_idx.size)] = flatten_dOut

        self._dinX = self._col2im(self._dinX, self._inX_padded.shape)
        self._dinX.reshape(self._inX.shape)

        return self._dinX