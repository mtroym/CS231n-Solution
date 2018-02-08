from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        (C, H, W) = input_dim
        sigma = weight_scale
        self.params['W1'] = sigma * np.random.randn(num_filters,C,filter_size,filter_size)
        self.params['b1'] = np.zeros((1,num_filters))
        self.params['W2'] = sigma * np.random.randn(int(num_filters*W*H/4),hidden_dim)
        self.params['b2'] = np.zeros((1,hidden_dim))
        self.params['W3'] = sigma * np.random.randn(hidden_dim,num_classes)
        self.params['b3'] = np.zeros((1,num_classes))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        # . - affine - softmax
        layer_input = X
        layer_input,cache_conv = conv_forward_fast(layer_input,W1,b1,conv_param)
        layer_input,cache_relu1 = relu_forward(layer_input)
        layer_input,cache_pool = max_pool_forward_fast(layer_input,pool_param)
        layer_input,cache_aff1 = affine_forward(layer_input,W2,b2)
        layer_input,cache_relu2 = relu_forward(layer_input)
        scores,cache_aff2 = affine_forward(layer_input,W3,b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss,daff2 = softmax_loss(scores,y)
        loss +=  0.5*self.reg*(np.sum(W3*W3)+np.sum(W2*W2)+np.sum(W1*W1))
        dlast_layer,dW3,db3 = affine_backward(daff2,cache_aff2)
        dlast_layer = relu_backward(dlast_layer,cache_relu2)
        dlast_layer,dW2,db2 = affine_backward(dlast_layer,cache_aff1)
        dlast_layer = max_pool_backward_fast(dlast_layer,cache_pool)
        dlast_layer = relu_backward(dlast_layer,cache_relu1)
        dlast_layer,dW1,db1 = conv_backward_fast(dlast_layer,cache_conv)
        grads['W1'], grads['b1'] = dW1+self.reg*W1,db1
        grads['W2'], grads['b2'] = dW2+self.reg*W2,db2
        grads['W3'], grads['b3'] = dW3+self.reg*W3,db3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
