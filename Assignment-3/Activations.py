import numpy as np

# ----------SIGMOID---------------

def sigmoid(z):
    z = np.clip(z , -500 , 500)
    #this is the non-linear activation function
    # print(z)
    return (1 / (1 + np.exp(-z)))

def sigmoid_back(z):
    z = np.clip(z , -500 , 500)
    # d sigmoid(z) / dz = sigmoid(z) [ 1 - sigmoid(z)]
    return sigmoid(z)* ( 1 - sigmoid(z))


# ------------TANH--------------

def tanh(z):
    return np.tanh(z)

def tanh_back(z):
    return (1 - tanh(z) * tanh(z))

# ------------RELU------------

def relu(z):
    return (z > 0 ) * z

def relu_back(z):
    return (z>0).astype(np.float64)

# ------------SOFTMAX-------------
def softmax(z):
    '''
    This is assuming softmax regularization on axis = 1(normalizing along the final layer)
    Ony for 2 dimensional input

    '''

    z = np.clip(z , -800 , 800)

    z = np.asarray(z).T
    z = z - z.max(0)
    exp_z = np.exp(z)
    softmax_z = (exp_z / exp_z.sum(0)).T
    return softmax_z

def softmax_back(z):
    # z = np.asarray(z).T
    # z = z - z.max(0)
    # exp_z = np.exp(z)
    # softmax_z = (exp_z / exp_z.sum(0)).T

    return np.ones_like(z)

# -----------LINEAR----------------
def linear(z):
    return z

def linear_back(z):
    # returning df/dz
    return np.ones_like(z)

#--------FUNCTIONS TO CALL -----------
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

    elif identifier == 'relu':
        return relu
    elif identifier == 'softmax':
        return softmax

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

    elif identifier == 'relu':
        return relu_back
    elif identifier == 'softmax':
        return softmax_back

    elif identifier == 'linear':
        return linear_back
    else:
        raise Exception('The {} activation function is not implemented'.format(identifier))
