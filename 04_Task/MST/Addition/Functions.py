
import numpy as np


def default_random_normal_dist__(size : tuple):
    return np.random.normal(loc=0, scale=1, size = size)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))