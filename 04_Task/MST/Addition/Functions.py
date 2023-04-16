
import numpy as np


def default_random_normal_dist__(size : tuple):
    return np.random.normal(loc=0, scale=1, size = size)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softMax(x):
    # Вычитаем из всех X - максимум по X, чтобы уменьшить переполнение экспоненты
    x_exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    outX = x_exp / np.sum(x_exp, axis=1, keepdims=True)

    outX = np.nan_to_num(outX)
    return outX