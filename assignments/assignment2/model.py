import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        
        self.fcc = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.fcc_hidden = FullyConnectedLayer(hidden_layer_size, n_output)
        #self.relu_hidden = ReLULayer()

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        self.fcc.W.grad = None
        self.fcc.B.grad = None
        self.fcc_hidden.W.grad = None
        self.fcc_hidden.B.grad = None
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        ll = self.fcc.forward(X)
        rel = self.relu.forward(ll)
        ll_hidden = self.fcc_hidden.forward(rel)
        #rel_hidden = self.relu_hidden.forward(ll_hidden)
        
        
        
        loss, softmax_grad = softmax_with_cross_entropy(ll_hidden, y)
        #relu_hid_grad = self.relu_hidden.backward(softmax_grad)
        fcc_hidden_grad = self.fcc_hidden.backward(softmax_grad)
        relu_grad = self.relu.backward(fcc_hidden_grad)
        fcc_grad = self.fcc.backward(relu_grad)
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        w_fcc_l2_loss, w_fcc_l2_grad = l2_regularization(self.fcc.W.value, self.reg)
        b_fcc_l2_loss, b_fcc_l2_grad = l2_regularization(self.fcc.B.value, self.reg)
        w_fcc_hidden_l2_loss, w_fcc_hidden_l2_grad = l2_regularization(self.fcc_hidden.W.value, self.reg)
        b_fcc_hidden_l2_loss, b_fcc_hidden_l2_grad = l2_regularization(self.fcc_hidden.B.value, self.reg)
        
        loss += w_fcc_l2_loss + w_fcc_hidden_l2_loss + b_fcc_l2_loss + b_fcc_hidden_l2_loss
        
        # TODO Now implement l2 regularization in the forward and backward pass
        #добавляем в градиенты по всем параметрам соответствующие градиенты по L2 регуляризации
        self.fcc.W.grad += w_fcc_l2_grad
        self.fcc.B.grad += b_fcc_l2_grad
        self.fcc_hidden.W.grad += w_fcc_hidden_l2_grad
        self.fcc_hidden.B.grad += b_fcc_hidden_l2_grad

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        #pred = np.zeros(X.shape[0], np.int)

        #raise Exception("Not implemented!")
        ll = self.fcc.forward(X)
        rel = self.relu.forward(ll)
        ll_hidden = self.fcc_hidden.forward(rel)
        #rel_hidden = self.relu_hidden.forward(ll_hidden)
        probs = softmax(ll_hidden)
        pred = np.argmax(probs, axis = 1)
        
        return pred

    def params(self):
        # TODO Implement aggregating all of the params

        result = {
            'fcc_w':self.fcc.W,
            'fcc_b':self.fcc.B,
            'fcc_hidden_w':self.fcc_hidden.W,
            'fcc_hidden_b':self.fcc_hidden.B
        }

        return result
