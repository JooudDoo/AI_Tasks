
import MST

def ReLU(val : MST.MDT_REFACTOR_ARRAY) -> MST.MDT_REFACTOR_ARRAY:
    return MST.ReLU()(val)

def Sigmoid(val : MST.MDT_REFACTOR_ARRAY) -> MST.MDT_REFACTOR_ARRAY:
    return MST.Sigmoid()(val)

def Flatten(val : MST.MDT_REFACTOR_ARRAY):
    return MST.Flatten()(val)