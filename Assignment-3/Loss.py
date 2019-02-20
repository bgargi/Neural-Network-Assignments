import numpy as np


class Loss:
    '''
    Class containing loss functions
    '''

    def __init__(self):
        return

    def sq_loss(self , pred ,Y):
        loss = Y - pred
        # d_loss / d_pred
        d_pred = -1 * (loss)

        loss = 0.5 * np.sum(loss * loss)

        return loss,d_pred
