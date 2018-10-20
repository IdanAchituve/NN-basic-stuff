import numpy as np

STUDENT={'name': 'Idan Achituve',
         'ID': '300083029'}

def classifier_output(x, params, ber_masks=None):
    import mlp1

    num_params = len(params) # get the length of params vector

    if ber_masks is None:  # if there isn't a bernouli mask all masks are matrices with 1 in all elements
        ber_masks = []
        for i in xrange(1,num_params-2,2):
            ber_masks.append(np.ones(params[i].shape[0])) # the mask size is with the sahpe of b


    activations = np.copy(x)
    idx = 0
    # in case the network has no hidden layers, no activation is needed and no iterations of the loop will be held
    for i in xrange(0,num_params-2,2): # calculate the activation of each subsequent layers and apply the bernouly mask
        params_w_b = []
        params_w_b.append(params[i])
        params_w_b.append(params[i + 1])
        activations = mlp1.classifier_activation(activations,params_w_b)
        activations = np.multiply(activations, ber_masks[idx])  # apply dropout on hidden layer based on corresponding bernouly vector
        idx+=1

    import loglinear as ll
    W = params[num_params-2]
    b = params[num_params-1]
    probs = ll.classifier_output(activations,[W,b]) # compute probability of output neurons according to softmax
    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params, reg_lambda=None, ber_masks=None):

    import math as m
    num_params = len(params) # number of parameters

    if reg_lambda is None: # if there isn't a regularization assign 0 to lambda
        reg_lambda = 0
    if ber_masks is None: # if there isn't a bernouli mask all masks are matrices with 1 in all elements
        ber_masks = []
        for i in xrange(1,num_params-2,2):
            ber_masks.append(np.ones(params[i].shape[0])) # the mask size is with the sahpe of b

    y_hat = classifier_output(x, params, ber_masks)  # get probability vector
    loss = -m.log(y_hat[y], m.e) # compute the loss wo regularization
    for i in xrange(0,num_params,2): # add regularization of all layers weights
        loss += reg_lambda * np.sum(np.power(params[i], 2))

    y_dim = y_hat.shape  # get dimensions of y
    y_vec = np.zeros(y_dim)  # create 1-hot-vector
    y_vec[y] = 1

    # backpropagation:
    grads = []
    import mlp1

    # compute delta of last layer and the corresponding gradeints
    if len(ber_masks) == 0: # no hidden layers
        gW = np.outer(x, np.subtract(y_hat, y_vec))  # compute dL/dw[i,j] according to (y_hat[j]-y[j])*x[i]
        gb = np.subtract(y_hat, y_vec)  # compute dL/db[j] according to y_hat[j]-y[j]
        grads.append(gW)
        grads.append(gb)
    else:
        activations = []
        activations.append(np.copy(x))
        idx = 0
        for i in xrange(0,num_params-2,2):  # calculate the activation of each subsequent layers and apply the bernouly mask
            a = activations[idx]
            params_b_w = []
            params_b_w.append(params[i])
            params_b_w.append(params[i+1])
            next_act = mlp1.classifier_activation(a, params_b_w) # compute activation based on parameter and previous layer
            next_act = np.multiply(next_act, ber_masks[idx])  # apply dropout on hidden layer based on corresponding bernouly vector
            idx+=1
            activations.append(next_act)

        delta = np.subtract(y_hat, y_vec)  # vectorized implantation of delta_out[i] = y_hat[i]-y[i]
        for i in reversed(xrange(0,num_params,2)):
            prev_layer_activions = activations[idx]
            if i < (num_params-2):
                # compute delta of the hidden layer and the corresponding gradeints
                da_dz = np.multiply(np.subtract(1, np.power(curr_layer_activations, 2)),ber_masks[idx])  # the derivative of the activation of the second layer by z
                delta = np.multiply(np.dot(W, delta),da_dz)  # vectorized implantation of delta_hid[i] = delta_out*W(.)(1-a^2)

            curr_layer_activations = prev_layer_activions
            W = params[i] # for next iteration delta calculations
            gW = np.outer(prev_layer_activions, delta) + 2 * reg_lambda * W  # vector-vector outer product multiplication
            gb = delta
            grads.append(gb) # add gradients to the list
            grads.append(gW) # add gradients to the list
            idx-=1

        grads = grads[::-1] # reverse the order of items in the list
    return loss, grads


def create_classifier(dims):

    params = []

    for first_layer, second_layer in zip(dims, dims[1:]):
        W = np.zeros((first_layer, second_layer))
        b = np.zeros(second_layer)
        params.append(W) # create parameters and append to end of list
        params.append(b) # create parameters and append to end of list

    return params