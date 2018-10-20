import numpy as np

STUDENT={'name': 'Idan Achituve',
         'ID': '300083029'}


def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    # YOUR CODE HERE

    x = np.exp(x - np.max(x))
    x /= np.sum(x)

    return x
    

def classifier_output(x, params):
    """
    Return the output layer (class probabilities) 
    of a log-linear classifier with given params on input x.
    """
    W,b = params

    probs = np.dot(x,W) # matrix vector multiplication
    probs = np.add(probs,b) # vector element wise summation
    probs = softmax(probs)

    return probs

def predict(x, params):
    """
    Returnss the prediction (highest scoring class id) of a
    a log-linear classifier with given parameters on input x.
    """
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    Compute the loss and the gradients at point x with given parameters.
    y is a scalar indicating the correct label.

    returns:
        loss,[gW,gb]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    """
    W,b = params
    # YOU CODE HERE

    import math as m

    x = np.array(x)
    y_hat=classifier_output(x, params) # get probability vector

    loss = -m.log(y_hat[y],m.e) # the loss for hard labels

    y_dim = y_hat.shape # get dimensions of y
    y_vec = np.zeros(y_dim) # create 1-hot-vector
    y_vec[y] = 1

    gW = np.outer(x,np.subtract(y_hat,y_vec)) # compute dL/dw[i,j] according to (y_hat[j]-y[j])*x[i]
    gb = np.subtract(y_hat,y_vec) # compute dL/db[j] according to y_hat[j]-y[j]

    return loss,[gW,gb]

def create_classifier(in_dim, out_dim):
    """
    returns the parameters (W,b) for a log-linear classifier
    with input dimension in_dim and output dimension out_dim.
    """
    W = np.zeros((in_dim, out_dim))
    b = np.zeros(out_dim)
    return [W,b]

def my_checks():

    import math as m

    # test softmax func:
    test_i =softmax(np.array([m.log(1, m.e), m.log(2, m.e), m.log(3, m.e), m.log(4, m.e)]))
    print test_i
    assert np.amax(np.fabs(test_i - np.array([0.1, 0.2, 0.3, 0.4]))) != 0

    # test classifier_output func:
    W = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    b = np.array([1, 2])
    x = np.array([1, 2, 3, 4])
    params = (W, b)
    probs = classifier_output(x, params)
    print(probs)

if __name__ == '__main__':

    # Sanity checks for softmax. If these fail, your softmax is definitely wrong.
    # If these pass, it may or may not be correct.

    # my_checks()

    test1 = softmax(np.array([1,2]))
    print test1
    assert np.amax(np.fabs(test1 - np.array([0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([1001,1002]))
    print test2
    assert np.amax(np.fabs(test2 - np.array( [0.26894142, 0.73105858]))) <= 1e-6

    test3 = softmax(np.array([-1001,-1002])) 
    print test3 
    assert np.amax(np.fabs(test3 - np.array([0.73105858, 0.26894142]))) <= 1e-6


    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W,b = create_classifier(3,4)

    def _loss_and_W_grad(W):
        global b
        loss,grads = loss_and_gradients([1,2,3],0,[W,b])
        return loss,grads[0]

    def _loss_and_b_grad(b):
        global W
        loss,grads = loss_and_gradients([1,2,3],0,[W,b])
        return loss,grads[1]

    for _ in xrange(10):
        W = np.random.randn(W.shape[0],W.shape[1])
        b = np.random.randn(b.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
        
