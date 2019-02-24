import numpy as np



def mean_square_error(pred,Y):
    N = pred.shape[0]
    loss = Y - pred

    # d_loss / d_pred
    d_pred = np.mean(-1 * (loss))

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
    d_pred = np.mean(-1 * mask)

    return loss,d_pred

def binary_cross_entropy(pred , Y):

    first_term = Y * np.log(pred)
    second_term = (1 - Y) * np.log(1 - pred)
    loss = -1  * np.sum( first_term + second_term ,axis =0)
    d_pred = np.mean(np.nan_to_num((pred - Y) / (pred * (1-pred))))

    return loss,d_pred


def multiclass_cross_entropy(pred, Y):
    num_classes = Y.shape[1]
    N = Y.shape[0]
    loss = (-1/N)*(np.log(pred)*Y).sum()#.reshape([N,1])
    d_pred = np.mean((-1*Y*np.nan_to_num(1/pred*float(N))).sum(1))
    #print("pred = ",pred)
    #print("Y = ", Y)
    #print("d_pred = ",d_pred)

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
