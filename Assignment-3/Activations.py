import numpy as np


"""
This file has the important activation functions.

The code to the MultiLayerPerceptron is extendible and any
new activation function can be easily added to this this file by
using the following format

### For the Front Propogation

```
def activationFunctionName(input):
    return activated_input
```

Note that the shape of inpt and activated input needs to be same

### For Back Propogation
```
def activationFunctionName_back(input):
    return d_back
```

 - d_back : represent the partial derivative of the function w.r.t.
        input. Shape of d_back is same as input

Note that you will need to include the new activation function string
to the get and get_back functions.
"""

# ----------SIGMOID---------------

def sigmoid(z):
    """
    Sigmoid Activation

    Inputs:
    - z : A 2D numpy array
    """
    z = np.clip(z , -500 , 500)
    return (1 / (1 + np.exp(-z)))

def sigmoid_back(z):
    """
    Backpropogation for Sigmoid Activation

    Inputs:
    - z : A 2D numpy array
    """
    z = np.clip(z , -500 , 500)
    # d sigmoid(z) / dz = sigmoid(z) [ 1 - sigmoid(z)]
    return sigmoid(z)* ( 1 - sigmoid(z))


# ------------TANH--------------

def tanh(z):
    """
    Tanh Activation

    Inputs:
    - z : A 2D numpy array
    """
    return np.tanh(z)

def tanh_back(z):
    """
    Backpropogation for Tanh Activation

    Inputs:
    - z : A 2D numpy array
    """
    return (1 - tanh(z) * tanh(z))

# ------------RELU------------

def relu(z):
    """
    Relu Activation

    Inputs:
    - z : A 2D numpy array
    """
    return (z > 0 ) * z

def relu_back(z):
    """
    Backpropogation for Relu Activation

    Inputs:
    - z : A 2D numpy array
    """
    return (z>0).astype(np.float64)

# ------------SOFTMAX-------------
def softmax(z):
    """
    Softmax Activation.This applies softmax normalization on axis =1
    Note that when using softmax use **the softmax_multiclass_cross_entropy loss only and
    this activation can be used in the last layer only**.(defined in [[Loss.py]])

    Inputs:
    - z : A 2D numpy array
    """
    z = np.clip(z , -800 , 800)

    z = np.asarray(z).T
    z = z - z.max(0)
    exp_z = np.exp(z)
    softmax_z = (exp_z / exp_z.sum(0)).T
    return softmax_z

def softmax_back(z):
    """
    Backpropogation for Softmax Activation.
    The partial differentiation is handeled with softmax_multiclass_cross_entropy

    Inputs:
    - z : A 2D numpy array
    """
    # z = np.asarray(z).T
    # z = z - z.max(0)
    # exp_z = np.exp(z)
    # softmax_z = (exp_z / exp_z.sum(0)).T

    return np.ones_like(z)

# -----------LINEAR----------------
def linear(z):
    """
    Linear Activation

    Inputs:
    - z : A 2D numpy array
    """
    return z

def linear_back(z):
    """
    Backpropogation for Linear Activation

    Inputs:
    - z : A 2D numpy array
    """
    # returning df/dz
    return np.ones_like(z)

#--------FUNCTIONS TO CALL -----------
def get(identifier = None):
    """
    This function gets fetches the activation function identified by
    the string 'identifier'.If such a function is not implemented
    this raises an Exception.


    Inputs:
    - identifier : a string to identify the activation function to fetch
    (default value = None.This fetches linear activaton)

    """

    if identifier == None:
        return linear

    elif identifier == 'sigmoid':
        return sigmoid

    elif identifier == 'tanh':
        return tanh

    elif identifier == 'relu':
        return relu
    elif identifier == 'softmax':
        return softmax

    elif identifier == 'linear':
        return linear
    else:
        raise Exception('The {} activation function is not implemented'.format(identifier))


def get_back(identifier):
    """
    This function gets fetches the functions for backpropogating identified by
    the string 'identifier'.If such a function is not implemented
    this raises an Exception.


    Inputs:
    - identifier : a string to identify the activation_back function to fetch
    (default value = None.This fetches linear backpropogation)

    """

    if identifier == None:
        return linear_back

    elif identifier == 'sigmoid':
        return sigmoid_back

    elif identifier == 'tanh':
        return tanh_back

    elif identifier == 'relu':
        return relu_back
    elif identifier == 'softmax':
        return softmax_back

    elif identifier == 'linear':
        return linear_back
    else:
        raise Exception('The {} activation function is not implemented'.format(identifier))
