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
        pad = (filter_size - 1) // 2
        stride = 1
        C, H, W = input_dim
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        output_H = 1 + (H + 2 * pad - filter_size) // stride
        output_W = 1 + (W + 2 * pad - filter_size) // stride
        self.params['W2'] = weight_scale * np.random.randn(num_filters * (output_H//2) * (output_W//2), hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)
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
        conv_out,conv_cache = conv_forward_fast(X,W1,b1,conv_param)
        relu_out,relu_cache = relu_forward(conv_out)
        pool_out,pool_cache = max_pool_forward_fast(relu_out,pool_param)
        pool_out_shape = pool_out.shape
        pool_out = pool_out.reshape((pool_out.shape[0],-1))
        affine_relu_out,affine_relu_cache = affine_relu_forward(pool_out,W2,b2)
        affine_out,affine_cache = affine_forward(affine_relu_out,W3,b3)
        scores = affine_out
        # out1, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        # # reshape the out1 for affine
        # out1_flat = out1.reshape((out1.shape[0], -1))
        # # out2: (N, hidden_dim)
        # out2, affine1_cache = affine_relu_forward(out1_flat, W2, b2)
        # # out3: (N, num_classes)
        # out3, affine2_cache = affine_forward(out2, W3, b3)

        # scores = out3
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
        loss,dscores = softmax_loss(scores,y)
        L2 = np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3)
        loss += 0.5 * self.reg * L2
        dout,dW3,db3 = affine_backward(dscores,affine_cache)
        dout,dW2,db2 = affine_relu_backward(dout,affine_relu_cache)
        dout = dout.reshape(pool_out_shape)
        dout = max_pool_backward_fast(dout,pool_cache)
        dout = relu_backward(dout,relu_cache)
        dout,dW1,db1 = conv_backward_fast(dout,conv_cache)
        dW1 += 0.5*self.reg*W1
        dW2 += 0.5*self.reg*W2
        dW3 += 0.5*self.reg*W3
        grads['W1'], grads['b1'] = dW1, db1
        grads['W2'], grads['b2'] = dW2, db2
        grads['W3'], grads['b3'] = dW3, db3
        # # dscores: (N, num_classes)
        # loss, dscores = softmax_loss(scores, y)
        # loss += self.reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))
        # # dx3: (N, hidden_dims)
        # dx3, dW3, db3 = affine_backward(dscores, affine2_cache)
        # # dx2: (N, num_filters * filter_size / 2 * filter_size / 2 )
        # dx2, dW2, db2 = affine_relu_backward(dx3, affine1_cache)
        # #print(dx2.shape)
        # #print(W2.shape)
        # dx, dW1, db1 = conv_relu_pool_backward(dx2.reshape(out1.shape), conv_cache)
        # dW1 += 2 * self.reg * W1
        # dW2 += 2 * self.reg * W2
        # dW3 += 2 * self.reg * W3
        # grads['W1'], grads['b1'] = dW1, db1
        # grads['W2'], grads['b2'] = dW2, db2
        # grads['W3'], grads['b3'] = dW3, db3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
