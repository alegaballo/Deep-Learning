# transfer functions


import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivative of sigmoid

def dsigmoid(y):
    return y * (1.0 - y)
 
    
    
   
    
    
def tanh(x):
    return np.tanh(x)

# derivative for tanh sigmoid

def dtanh(y):
    return 1 - y*y