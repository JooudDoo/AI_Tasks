
import numpy as np

from MST import MDT_REFACTOR_ARRAY, MDT_ARRAY
from MST.Addition import HE_weight_init__

from .Basic import ConvBasicModule

class Conv2d(ConvBasicModule):

    def __init__(self, in_channels : int, out_channels : int, kernel_size, stride = 1, padding = 0, dilation = 1, use_bias = True, padding_value : float = 0):
        super().__init__(kernel_size, stride, padding, dilation, padding_value)
        self._inC = in_channels
        self._outC = out_channels

        self._use_bias = use_bias

        self.__init_weights()

    def __init_weights(self):
        self._w = HE_weight_init__(size=(self._outC, self._inC, *self._kernel_size))
        self._bias = np.zeros(shape=(1, self._outC, 1, 1), dtype=np.float32) #HE_weight_init__(size=(1, self._outC, 1, 1)) if self._use_bias else 
    
    def forward(self, x):
        BS, C, H, W = x.shape
        self._output_size = self._calculate_output_sizes(H, W)

        self._inX = np.pad(x, [(0,0), (0,0), (self._padding[0], self._padding[0]), (self._padding[1], self._padding[1])], mode='constant', constant_values=self._padding_value)

        self._inX_cols = self._im2col(self._inX) # .shape() = [C * K * K, BS * outH * outW]
        self._flatten_w = self._w.reshape(self._outC, -1) # .shape() = [outC, inC * K * K]

        self._outX = np.dot(self._flatten_w, self._inX_cols).reshape(self._outC, *self._output_size, BS)
        self._outX = self._outX.transpose(3, 0, 1, 2)

        if self._use_bias:
            self._outX += np.tile(self._bias, (BS, 1, *self._output_size))
        return self._outX
    
    def backward_impl(self, dOut=None):
        flatten_dOut = dOut.reshape(self._outC, -1) # [outC, outH * outW * BS]
        # self._inX_cols.T .shape = [BS * outH * outW, C * K * K]
        self._dw = np.dot(flatten_dOut, self._inX_cols.T).reshape(self._w.shape) # outC, C * K * K

        self._dinX = np.dot(self._flatten_w.T, flatten_dOut) # [outC, K * K * C].T * [outC, outH * outW * BS] -> [K*K*C, outH * outW * BS]
        self._dinX = self._col2im(self._dinX, self._inX.shape) # col2im -> [BS, inC, outH, outW]

        if self._use_bias:
            self._dbias = np.sum(dOut, axis=(0, 2, 3), keepdims=True)

        return self._dinX