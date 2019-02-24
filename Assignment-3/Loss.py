import numpy as np



def mean_square_error(pred,Y):
    N = pred.shape[0]
    loss = Y - pred

    # d_loss / d_pred
    d_pred = (np.multiply(-1,loss)) / float(N)

    loss = 0.5 * np.mean(np.square(loss) , axis = 0)

    return loss,d_pred

def mean_abs_error(pred , Y):
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
    '''
    pred should be in range [0,1]
    '''
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
    epsilon = 1e-11
    pred = np.clip(pred , epsilon , 1 - epsilon)
    loss = np.mean(-np.sum(np.log(pred)*Y, axis = 1) , axis = 0)
    d_pred = pred - Y
    return loss,d_pred

def get(identifier):

    '''
    '''

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
