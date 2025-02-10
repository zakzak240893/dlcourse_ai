import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    predictions = predictions.copy()
    if np.ndim(predictions) == 1:
        predictions -= np.max(predictions)
        return np.exp(predictions) /np.sum(np.exp(predictions), axis = 0)
    else:
        predictions -= np.max(predictions, axis = 1).reshape(-1,1)
        return np.exp(predictions) /np.sum(np.exp(predictions), axis = 1).reshape(-1,1)
    
def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    return np.mean(
        -np.log(
            probs[np.arange(probs.shape[0]), target_index.reshape(-1)]
        )
    )

def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength * np.sum(np.power(W,2))
    grad = reg_strength * 2 * W

    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    sm = softmax(preds)
    
    loss = cross_entropy_loss(sm, target_index)
    
    mask = np.zeros_like(sm, dtype = float)
    if np.ndim(preds) == 1:
        
        mask[target_index] = 1
        dprediction = sm
        dprediction -=mask 
    else:
        mask[np.arange(mask.shape[0]), target_index.reshape(-1)] = 1
        dprediction = (sm - mask) / sm.shape[0]    
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    
    
    
    def __init__(self):
        self.output = None
        self.grad = None
        self.x = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        #raise Exception("Not implemented!")
        output = (X + np.abs(X)) / 2
        self.output = output
        self.x = X
        return output
        

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        return (self.x > 0) * d_out

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        #self.W = Param( 0.001 * np.random.randn(n_input, n_output))
        self.W = Param(1. / np.sqrt(n_input / 2.) * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        result =  np.dot(X, self.W.value) + self.B.value
        return result

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        
        self.W.grad = np.dot(self.X.T, d_out)
        
        self.B.grad = np.dot(np.ones((self.X.shape[0],1)).T, d_out)
        #print(self.X.shape, self.B.grad.shape, np.ones((self.X.shape[0], 1)).T.shape, d_out.shape)
        
        d_result = np.dot(d_out,self.W.value.T)

        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}
