import numpy as np

def sigmoid(z):
    #this is the non-linear activation function
    # print(z)
    return (1 / (1 + np.exp(-z)))

def sigmoid_back(z):
    # d sigmoid(z) / dz = sigmoid(z) [ 1 - sigmoid(z)]
    return sigmoid(z)* ( 1 - sigmoid(z))

def tanh(z):
    return np.tanh(z)

def tanh_back(z):
    return (1 - tanh(z) * tanh(z))


def linear(z):
    return z

def linear_back(z):
    # returning df/dz
    return 1

#
def get(identifier):
    '''
    Gets the 'identifier' activation functions


    if 'identifier' == None returns linear
    '''

    if identifier == None:
        return linear

    elif identifier == 'sigmoid':
        return sigmoid

    elif identifier == 'tanh':
        return tanh

    elif identifier == 'linear':
        return linear
    else:
        raise Exception('The {} activation function is not implemented'.format(identifier))


def get_back(identifier):
    '''
    Gets the 'identifier' activation functions


    if 'identifier' == None returns linear
    '''

    if identifier == None:
        return linear_back

    elif identifier == 'sigmoid':
        return sigmoid_back

    elif identifier == 'tanh':
        return tanh_back

    elif identifier == 'linear':
        return linear_back
    else:
        raise Exception('The {} activation function is not implemented'.format(identifier))
