import numpy as np

def default_random_normal_dist__(size : tuple):
    return np.random.normal(loc=0, scale=1, size = size)

# !TODO добавить параметр slope для разных типов активации (sigmoid/leakyReLU...)
def HE_weight_init__(size : tuple, dtype = np.float32):
    """
        Нулевое среднее
        Но дисперсия должна быть с учетом размера входных параметров
    """
    outS = size[1] # выходной размер после применения весов
    inS = size[0] # входной размер весов
    receptive = 1
    if(len(size) > 2):
        for d in size[2:]:
            receptive *= d
    disp = np.sqrt(2/(outS*receptive))
    return np.random.normal(loc=0, scale=disp, size=size).astype(dtype)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softMax(x):
    # Вычитаем из всех X - максимум по X, чтобы уменьшить переполнение экспоненты
    x_exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    outX = x_exp / np.sum(x_exp, axis=1, keepdims=True)

    outX = np.nan_to_num(outX)
    return outX