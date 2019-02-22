import numpy as np



def mean_square_error(pred,Y):
    N = pred.shape[0]
    loss = Y - pred

    # d_loss / d_pred
    d_pred = (-1 * (loss)) / float(N)

    loss = 0.5 * np.mean(loss * loss , axis = 0)

    return loss,d_pred

def mean_abs_error(pred , Y):
    N = pred.shape[0]
    loss = Y -pred
    # mask to account for the absolute function
    mask = (loss >= 0)
    mask = np.array((mask * 2) - np.ones_like(mask))

    loss = np.mean(np.abs(Y - pred) , axis = 0)
    #d_loss / d-pred
    d_pred = (-1 * mask) / float(N)

    return loss,d_pred

def binary_cross_entropy(pred , Y):

    first_term = Y * np.log(pred)
    second_term = (1 - Y) * np.log(1 - pred)
    loss = -1  * np.sum( first_term + second_term ,axis =0)
    d_pred = (pred - Y) / (pred * (1-pred))

    return loss,d_pred

def mean_binary_cross_entropy(pred , Y):
    N = pred.shape[0]

    first_term = Y * np.log(pred)
    second_term = (1 - Y) * np.log(1 - pred)
    loss = -1  * np.mean( first_term + second_term ,axis =0)
    d_pred = (pred - Y) / (float(N) *pred * (1-pred))

    return loss,d_pred

def mean_multiclass_cross_entropy(pred, Y):
    num_classes = Y.shape[1]
    N = Y.shape[0]
    loss = (-1/N)*(np.log(pred)*y).sum(1).reshape([N,1])
    d_pred = (-1*y*(1/p*float(N))).sum(0).reshape([num_classes,1])

    return loss, d_pred

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
    elif identifier=='mean_binary_cross_entropy':
        return mean_binary_cross_entropy
    elif identifier=='mean_multiclass_cross_entropy':
        return mean_multiclass_cross_entropy
    else:
        raise Exception('The {} loss function is not implemented'.format(identifier))
