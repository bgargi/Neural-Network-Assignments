import numpy as np
import Activations


class PerceptronLayer:
    '''
    This is a very specefic Neuron class that uses sigmoid activation
    and square loss.
    '''
    def __init__(self,l0,l1,activation=None):
        '''

        '''
        self.W = np.random.randn(l0 ,l1).astype(np.float64) * np.sqrt(2.0 /l0 )
        self.b = np.zeros((l1)).astype(np.float64)

        # self.grad = {'W':np.zeros((l0,l1)).astype(np.float64) ,
        #             'b':np.zeros((l1)).astype(np.float128) }

        self.act_fn = Activations.get(activation)
        self.act_fn_back = Activations.get_back(activation)


        print(self.W.shape , self.b.shape)
        print(self.act_fn)

    def forward(self,X):
        #print(X.shape , self.W.shape , self.b.shape)
        h_x = X.dot(self.W) + self.b

        a = self.act_fn(h_x)
        # print(a)
        return a,h_x



    def update_batch_gradient_descent(self , X , h_x , d_back,alpha = 0.01):
        '''
        Describe each variable
        '''

        # dloss/d(a) * d(a) / d(h_x)

        d_h_x = self.act_fn_back(h_x) * d_back
        #Derivating w.r.t W
        # dloss/d(h_x) * d(h_x) / d(W)
        # print((d_h_x).shape)

        d_W = X.T.dot(d_h_x)

        #Derivatng w.r.t b
        # dloss/d(h_x) * d(h_x) / d(b)
        d_b = d_h_x

        #Derivating w.r.t x(to return to previous layers)
        # dloss/d(h_x) * d(h_x) / d(X)
        d_X = d_h_x.dot(self.W.T)

        #Update W
        delta_W = d_W
        self.W = self.W - alpha * delta_W

        #Update b
        delta_b = np.sum(d_b , axis =0)
        self.b = self.b - alpha * delta_b

        d_back = d_X

        return d_back

class MultiLayerPerceptron:
    '''


    '''

    def __init__(self ,layer_list = None,activation_list = None):

        if layer_list == None:
            raise Exception('layer_list cant be empty')
        if activation_list == None:
            activation_list = []
        self.hidden_layers= len(layer_list) - 1

        self.layers = {}
        act_list_len = len(activation_list)
        for i in range(self.hidden_layers):
            if i + 1 <= act_list_len:
                self.layers[i+1] = PerceptronLayer(layer_list[i] , layer_list[i+1] , activation=activation_list[i])
            else:
                self.layers[i+1] = PerceptronLayer(layer_list[i] , layer_list[i+1])


    def forward(self , X_train):
        a = X_train
        cache = []
        cache.append(a)
        for i in range(self.hidden_layers):
            a,h_x = self.layers[i+1].forward(a)
            cache.append(h_x)

        return a,cache

    def update_gradient(self , cache, d_back , alpha=0.01):
        for i in range(self.hidden_layers , 0 , -1):
            d_back = self.layers[i].update_batch_gradient_descent(cache[i-1],cache[i],d_back,alpha)

        return d_back
