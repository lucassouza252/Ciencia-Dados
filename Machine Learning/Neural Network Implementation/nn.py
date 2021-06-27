# -*- coding: utf-8 -*-
import numpy as np
import random
 
class Network(object):
    """
        weights: Numpy Matrix storing numpy array of weights between the layers.
        biases: Numpy Matrix storing the bias
    """
    
    def __init__(self, sizes):
        """
        Boots bias and weight with random gaussian numbers

        Parameters
        ----------
        sizes : List with number of neurons by layer. Ex: [2, 3, 1] -> 2 input,
        3 hidden, 1 output.
        
        Returns
        -------
        None.

        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        
    
    def sigmoid(self, z):
        """
        Activation Function Sigmoid.

        Parameters
        ----------
        z : Value of (w*a + b) (weights * activations from second layers + bias).

        Returns
        -------
        Sigmoid value between 0 and 1.

        """
        return 1.0/(1.0 + np.exp(-z))
    
    def feedfoward(self, a):
        """
        FeedFoward function that apply the equation a' = sigmoid(w*a + b).

        Parameters
        ----------
        a : Values of Activations from second layers.

        Returns
        -------
        a : Values activated.

        """
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Stochastic Gradient Descent Function for train the Network

        Parameters
        ----------
        training_data : List of tuple with x and y data (train and target).
        epochs : Epochs for training.
        mini_batch_size : Mini batch to be used in sampling
        eta : Learning rate
        test_data : Test to evaluate the network by epoch.

        Returns
        -------
        None.

        """
        training_data = list(training_data)
        n = len(training_data)
        
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batchs = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch, eta)
                
            if test_data:
                print("Epoch {} : {} / {}".format(j, self.evaluate(test_data),n_test))
            else:
                print("Epoch {} finished".format(j))
    
    def update_mini_batch(self, mini_batch, eta):
        """
        

        Parameters
        ----------
        mini_batch : TYPE
            DESCRIPTION.
        eta : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        self.weights = [w-(eta/len(mini_batch)) + nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [d-(eta/len(mini_batch)) + nb for d, nb in zip(self.biases, nabla_b)]
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        