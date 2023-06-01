
import MST

def Relu(val : MST.MDT_REFACTOR_ARRAY) -> MST.MDT_REFACTOR_ARRAY:
    return MST.Relu()(val)

def sigmoid(val : MST.MDT_REFACTOR_ARRAY) -> MST.MDT_REFACTOR_ARRAY:
    return MST.Sigmoid()(val)

def flatten(val : MST.MDT_REFACTOR_ARRAY):
    return MST.Flatten()(val)

def sum(x_1 : MST.MDT_REFACTOR_ARRAY, x_2 : MST.MDT_REFACTOR_ARRAY):
    return MST.Sum()(x_1, x_2)