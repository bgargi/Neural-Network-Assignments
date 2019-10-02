import numpy as np

"""
This file defines various loss functions.
The function are defined in the following format:-
(Note that any function can be added as long as it follows the format)

def LossFunctionName(pred , Y):
	any operations
	return loss,d_pred

Inputs
-pred: a numpy array reprsenting predictions made by the model
-Y : a numpy  array representing the target variable

Note that shape(pred) = shape(Y)

Returns
- loss : a scalar value to represent the loss
- d_pred : a vector of the same shape as pred. It represents the error
        to be used for backpropogation

Note that you will need to include the new loss function string
to the get function.
"""

def mean_square_error(pred,Y):
    """
    The mean square error loss function


    Inputs
    -pred: a numpy array reprsenting predictions made by the model
    -Y : a numpy  array representing the target variable

    Note that shape(pred) = shape(Y)

    Returns
    - loss : a scalar value to represent the loss
    - d_pred : a vector of the same shape as pred. It represents the error
            to be used for backpropogation(shape(d_pred) = shape(pred))
    """
    N = pred.shape[0]
    loss = Y - pred

    # d_loss / d_pred
    d_pred = (np.multiply(-1,loss)) / float(N)

    loss = 0.5 * np.mean(np.square(loss) , axis = 0)

    return loss,d_pred

def mean_abs_error(pred , Y):
    """
    The mean absolute error loss function


    Inputs
    -pred: a numpy array reprsenting predictions made by the model
    -Y : a numpy  array representing the target variable

    Note that shape(pred) = shape(Y)

    Returns
    - loss : a scalar value to represent the loss
    - d_pred : a vector of the same shape as pred. It represents the error
            to be used for backpropogation(shape(d_pred) = shape(pred))
    """
    N = pred.shape[0]
    loss = Y -pred
    # mask to account for the absolute function
    mask = (loss >= 0)
    mask = np.array((mask * 2) - np.ones_like(mask))

    loss = np.mean(np.abs(Y - pred) , axis = 0)
    #d_loss / d-pred
    d_pred = (-1 * mask)/float(N)

    return loss,d_pred

def binary_cross_entropy(pred , Y):
    """
    The binary cross entropy loss.

    Note
    -pred should be in range [0,1]
    -Y should be a binary vector

    Inputs
    -pred: a numpy array reprsenting predictions made by the model
    -Y : a numpy  array representing the target variable

    Note that shape(pred) = shape(Y)

    Returns
    - loss : a scalar value to represent the loss
    - d_pred : a vector of the same shape as pred. It represents the error
            to be used for backpropogation(shape(d_pred) = shape(pred))
    """

    # clipping the inputs so there is no overflow
    epsilon = 1e-11
    pred = np.clip(pred , epsilon, 1 - epsilon)
    divisor = np.maximum(pred * (1-pred),epsilon)

    N = Y.shape[0]

    first_term = Y * np.log(pred)
    second_term = (1 - Y) * np.log(1 - pred)
    loss = -1  * np.mean( first_term + second_term ,axis =0)
    d_pred = np.nan_to_num((pred - Y) / (divisor * float(N)))

    return loss,d_pred


def multiclass_cross_entropy(pred, Y):
    """
    The multiclass cross entropy loss.Prefer using the
    softmax_multiclass_cross_entropy

    Note
    -pred should be in range [0,1]
    -Y should be Column vector of one hot row vectors

    Inputs
    -pred: a numpy array reprsenting predictions made by the model
    -Y : a numpy  array representing the target variable

    Note that shape(pred) = shape(Y)

    Returns
    - loss : a scalar value to represent the loss
    - d_pred : a vector of the same shape as pred. It represents the error
            to be used for backpropogation(shape(d_pred) = shape(pred))
    """

    # clipping the inputs so there is no overflow
    epsilon = 1e-11
    pred = np.clip(pred , epsilon, 1 - epsilon)

    num_classes = Y.shape[1]
    N = Y.shape[0]
    loss = np.mean(-np.sum(np.log(pred)*Y, axis = 1) , axis = 0)#.reshape([N,1])
    # print("pred = ",pred)
    # print("Y = ", Y)
    d_pred = (-1*Y*np.nan_to_num(1/(pred * float(N)))).sum(1).reshape(N,1)
    #print("pred = ",pred)
    #print("Y = ", Y)
    # print("d_pred = ",d_pred)

    return loss, d_pred

def softmax_multiclass_cross_entropy(pred ,Y):
    """
    The multiclass cross entropy loss with a softmax activation.

    Note
    -Use this loss when softmax(defined in [[Activations.py]]) is used as activation for the last layer.
    -pred should be in range [0,1]
    -Y should be Column vector of one hot row vectors

    Inputs
    -pred: a numpy array reprsenting predictions made by the model
    -Y : a numpy  array representing the target variable

    Note that shape(pred) = shape(Y)

    Returns
    - loss : a scalar value to represent the loss
    - d_pred : a vector of the same shape as pred. It represents the error
            to be used for backpropogation(shape(d_pred) = shape(pred))
    """
    epsilon = 1e-11
    pred = np.clip(pred , epsilon , 1 - epsilon)
    loss = np.mean(-np.sum(np.log(pred)*Y, axis = 1) , axis = 0)
    d_pred = pred - Y
    return loss,d_pred

def get(identifier):
    """
    This function gets fetches the loss identified by
    the string 'identifier'.If such a function is not implemented
    this raises an Exception.


    Inputs:
    - identifier : a string to identify the loss function to fetch
    (default value = None.This fetches MSE loss)

    """
    if identifier == None:
        return mean_square_error
    elif identifier == 'mean_square_error':
        return mean_square_error
    elif identifier == 'mean_abs_error':
        return mean_abs_error
    elif identifier=='binary_cross_entropy':
        return binary_cross_entropy
    elif identifier=='multiclass_cross_entropy':
        return multiclass_cross_entropy
    elif identifier=='softmax_multiclass_cross_entropy':
        return softmax_multiclass_cross_entropy
    else:
        raise Exception('The {} loss function is not implemented'.format(identifier))
