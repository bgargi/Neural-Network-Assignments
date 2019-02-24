import numpy as np
import Activations
import Loss
import Metrics

class PerceptronLayer:

    def __init__(self,l0,l1,activation=None):
        '''

        '''
        self.W = np.random.rand(l0 ,l1).astype(np.float64)# * np.sqrt(2.0 /l0 )
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
        #print("W_before = ", self.W)
        delta_W = d_W
        self.W = self.W - alpha * delta_W
        #print("W_after = ",self.W)

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


    def train(self,
            X_train,
            Y_train,
            X_test,
            Y_test,
            metric='accuracy_binary',
            loss_function_string = 'mean_square_error',
            epochs=200,
            record_at = 100,
            verbose = True,
            learning_rate =0.1,
            learning_rate_decay = False):

        loss_fn = Loss.get(loss_function_string)
        metric_fn = Metrics.get(metric)
        train_loss_his = []
        train_acc_his = []
        test_loss_his = []
        test_acc_his = []
        epoch_his = []

        for i in range(epochs):
            prediction , cache = self.forward(X_train)
            # ---------TO DO------
            # batch training to be incuded


            loss,d_back= loss_fn(prediction,Y_train)
            self.update_gradient(cache,d_back,learning_rate)
            if learning_rate_decay:
                learning_rate *= (1.0 / 1.0 + i)



            if i % record_at == 0:
                train_loss,_ = loss_fn(prediction,Y_train)
                train_acc = metric_fn(prediction,Y_train)

                test_prediction , _ = self.forward(X_test)
                test_loss,_ = loss_fn(test_prediction,Y_test)
                test_acc = metric_fn(test_prediction,Y_test)

                train_loss_his.append(train_loss)
                train_acc_his.append(train_acc)
                test_loss_his.append(test_loss)
                test_acc_his.append(test_acc)
                epoch_his.append(i)

                if verbose:
                    print("{}th EPOCH:\nTraining Loss:{}|Training Accuracy:{}|Test Loss:{}|Test Accuracy:{}".\
                      format(i , train_loss , train_acc,test_loss,test_acc))
        train_loss_his = np.array(train_loss_his).reshape(-1)
        train_acc_his = np.array(train_acc_his).reshape(-1)
        test_loss_his = np.array(test_loss_his).reshape(-1)
        test_acc_his = np.array(test_acc_his).reshape(-1)
        epoch_his = np.array(epoch_his).reshape(-1)
        return train_loss_his,train_acc_his,test_loss_his,test_acc_his,epoch_his

    def metric_function(self,X,Y,metric='accuracy_binary'):
        metric_fn = Metrics.get(metric)
        prediction , _ = self.forward(X)
        acc = metric_fn(prediction,Y)
        return acc
