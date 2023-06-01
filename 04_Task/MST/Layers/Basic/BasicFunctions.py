
import numpy as np

def im2col(image, output_size, kernel_size, stride):

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

def im2col_0(image, output_size, kernel_size, stride):
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

def col2im_0(cols, image_shape, output_size, kernel_size, stride, padding):
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

