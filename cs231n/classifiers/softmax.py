import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
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
  
  for i in range(X.shape[1]):
    scores = W.dot(X[:, i])
    scores -= np.max(scores) #prevents numerical instability
    prob = 0.0
    loss -= scores[y[i]]
    
    for curr_score in scores:
        prob += np.exp(curr_score)
    
    for j in range(W.shape[0]):
        prob_ji = np.exp(scores[j]) / prob
        margin = - prob_ji * X[:, i].T
        
        if j == y[i]:
            margin = (1 - prob_ji) * X[:, i].T
        dW[j, :] += -margin

    loss += np.log(prob)
         
  loss /= X.shape[1]
  dW /= X.shape[1]

  # Regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W


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
  
  scores = W.dot(X)
  scores += -np.max(scores) #prevents numerical instability

  correct_scores = scores[y, range(len(y))]
  exp_score = np.exp(scores)
  exp_sum = np.sum(exp_score,axis = 0) #adding up each column
  
  loss = - np.mean(np.log(np.exp(correct_scores) / exp_sum))
  
  n_probs = - exp_score / exp_sum
  n_probs[y, range(len(y))] += 1

  loss += 0.5 * reg * np.sum(W * W)
  
  dW = n_probs.dot(X.T)
  dW /= -X.shape[1]
  dW += reg * W

  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
