import numpy as np

STUDENT={'name': 'Idan Achituve',
         'ID': '300083029'}


def tanh(x):
    x = np.tanh(x)
    return x

# The following function compute the activation of neurons with the tanh function
def classifier_activation(x, params):

    W,b = params
    activations = np.dot(x, W)  # matrix vector multiplication
    activations = np.add(activations, b)  # vector element wise summation
    activations = tanh(activations)

    return activations

def classifier_output(x, params, ber_mask=None):

    import loglinear as ll

    if ber_mask is None:
        ber_mask = np.ones(params[1].shape[0])# bernouli for dropout (on prediction it is a vector of ones)

    activations = np.copy(x)

    params_w_b = [] # get the first 2 params
    params_w_b.append(params[0])
    params_w_b.append(params[1])
    activations = classifier_activation(activations, params_w_b) # compute the activation according to tanh function
    activations = np.multiply(activations,ber_mask) # apply dropout on hidden layer

    W = params[2] # weights to final layer
    b = params[3] # bias of final layer
    probs = ll.classifier_output(activations, [W, b])  # compute probability of output neurons according to softmax

    return probs


def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params, reg_lambda=None, ber_mask=None):


    import math as m

    if reg_lambda is None:  # if there isn't a regularization assign 0 to lambda
        reg_lambda = 0

    if ber_mask is None:
        ber_mask = np.ones(params[1].shape[0])  # bernouli for dropout (on prediction it is a vector of ones)

    y_hat = classifier_output(x, params, ber_mask)  # get probability vector
    loss = -m.log(y_hat[y], m.e) + reg_lambda*np.sum(np.power(params[0],2)) + reg_lambda*np.sum(np.power(params[2],2)) # the cross entropy loss for hard labels

    y_dim = y_hat.shape  # get dimensions of y
    y_vec = np.zeros(y_dim)  # create 1-hot-vector
    y_vec[y] = 1

    # backpropagation:

    # compute delta of last layer and the corresponding gradeints
    a = classifier_activation(x, [params[0],params[1]])  # compute the activation according to tanh function
    a = np.multiply(a, ber_mask)  # apply dropout on hidden layer
    delta_out = np.subtract(y_hat, y_vec)  # vectorized implantation of delta_out[i] = y_hat[i]-y[i]
    gW_out = np.outer(a, delta_out) + 2*reg_lambda*params[2] # vector-vector outer product multiplication
    gb_out = delta_out

    # compute delta of the hidden layer and the corresponding gradeints
    da_dz = np.multiply(np.subtract(1,np.power(a,2)),ber_mask) # the derivative of the activation of the second layer by z
    W = params[2]
    delta_hid = np.multiply(np.dot(W,delta_out),da_dz)  # vectorized implantation of delta_hid[i] = delta_out*W(.)(1-a^2)
    gW_hid = np.outer(x, delta_hid) + 2*reg_lambda*params[0]  # vector-vector outer product multiplication
    gb_hid = delta_hid

    grads = []
    grads.append(gW_hid)
    grads.append(gb_hid)
    grads.append(gW_out)
    grads.append(gb_out)

    return loss,grads

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.
    """
    params = []

    import loglinear as ll
    W1,b1 = ll.create_classifier(in_dim, hid_dim)
    W2,b2 = ll.create_classifier(hid_dim, out_dim)
    params.append(W1)
    params.append(b1)
    params.append(W2)
    params.append(b2)

    return params