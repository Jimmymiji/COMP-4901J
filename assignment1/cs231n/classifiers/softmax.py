import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X,W)
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_train):
    temp = scores[i,:]
    temp -= np.max(temp)
    sum_of_exponential = np.sum(np.exp(temp)) 
    p = np.exp(temp[y[i]]) /  sum_of_exponential 
    loss -= np.log(p)
    dW[:,y[i]] -= X[i] 
    for j in range(num_class):
      dW[:,j] += (np.exp(temp[j]) / sum_of_exponential)*X[i]
        
  loss /= num_train
  loss += reg*np.sum(W*W)
  dW /= num_train
  dW += reg*2*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X,W)
  num_train = X.shape[0]
  num_class = W.shape[1]
  scores -= np.amax(scores,axis=1).reshape(num_train,1)
  exp_scores = np.exp(scores)
  correct_scores = exp_scores[range(num_train),y]
  one_score_sums = np.sum(exp_scores,axis=1)
  loss = np.sum(-1*np.log(correct_scores/one_score_sums))
  loss = loss / num_train
  loss += reg*np.sum(W*W)
  one_score_sums = np.reshape(np.repeat(np.sum(exp_scores,axis=1),num_class),exp_scores.shape )
  probalibities = exp_scores / one_score_sums
  probalibities[range(num_train),y] = probalibities[range(num_train),y] - 1
  dW += np.dot(X.T, probalibities)
  dW = dW / num_train
  dW += 2*reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

