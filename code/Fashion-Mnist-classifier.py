import numpy as np
import math
from copy import copy

np.random.seed(111)


# import data using numpy loadtext method
def import_train_data():

    # get train data
    train_x = np.loadtxt("train_x")
    train_y = np.loadtxt("train_y")
    train_y = np.reshape(train_y, (np.shape(train_y)[0],1))

    # get dev data from train
    new_train_x, new_train_y, val_x, val_y = create_val_set(train_x, train_y)

    return new_train_x, new_train_y, val_x, val_y


# extract validation set from training set
def create_val_set(train_x, train_y):

    # generate an array of random numbers and create a mask
    rand_nums = np.random.rand(np.shape(train_y)[0])
    mask = rand_nums < 0.2

    # get validation data
    val_x = train_x[mask, :]
    val_y = train_y[mask]

    # get train data
    new_train_x = train_x[~mask, :]
    new_train_y = train_y[~mask]

    return new_train_x, new_train_y, val_x, val_y


# get the mean and std values according to the train set
def mean_and_std(train_x):

    mean = np.mean(train_x)
    std = np.std(train_x)

    return mean, std


# z-score normalization
def normalize_array(data, mean, std):
    return (data-mean)/std


# initialize params according to Glorot initialization
def intialize_weights(layers):

    params = []
    for layer, next_layer in zip(layers, layers[1:]):
        epsilon = np.sqrt(6.0/float(layer+next_layer))
        W = np.random.uniform(-epsilon, epsilon, (layer, next_layer))
        b = np.random.uniform(-epsilon, epsilon, (next_layer, 1))
        params.append(W)
        params.append(b)

    return params


# shuffle the training set
def randomize(x, y):

    # Generate the permutation index array.
    permutation = np.random.permutation(x.shape[0])

    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_x = x[permutation]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


# activation calculation during froward propagation
def activation(x, activation_func):

    if activation_func.lower() == "tanh":
        return np.tanh(x)
    elif activation_func.lower() == "sigmoid":
        return 1 / (1 + np.exp(-x))
    elif activation_func.lower() == "relu":
        return np.maximum(0, x)
    else: # softmax
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


# forward calculation
def forward(x, params, activation_func):

    w1, b1, w2, b2 = params

    # reshape x to a 2d matrix
    x = np.reshape(x, (np.shape(x)[0],1))

    # 1st layer calc
    z1 = np.dot(w1.T, x) + b1
    h1 = activation(z1, activation_func)

    # 2nd layer calc
    z2 = np.dot(w2.T, h1) + b2
    y_hat = activation(z2, "softmax")

    return y_hat, h1


# derivative of different activation functions
def derivative(x, activation_func):
    if activation_func.lower() == "tanh":
        return 1 - x**2
    elif activation_func.lower() == "sigmoid":
        return x * (1-x)
    elif activation_func.lower() == "relu":
        x[x <= 0] = 0
        x[x > 0] = 1
        return x


# calculate gradients for parameters
def calc_gradients(y, x, y_hat, h1, params, activation_func):

    w1, b1, w2, b2 = params

    # reshape x to a 2d matrix
    x = np.reshape(x, (np.shape(x)[0],1))

    # derivatives of the loss w.r.t last layer params
    dL_dw2 = np.dot(h1, (y_hat - y).T)
    dL_db2 = (y_hat - y)

    # derivatives of the loss w.r.t hidden layer params
    dL_dh1 = np.dot(w2, (y_hat - y))
    dL_dw1 = np.dot(x, (dL_dh1 * derivative(h1, activation_func)).T)
    dL_db1 = dL_dh1 * derivative(h1, activation_func)

    return dL_dw1, dL_db1, dL_dw2, dL_db2


# extract all weights parameters and merge them to one vector
def weights_to_vec(params):

    w1, b1, w2, b2 = params

    # reshape from matrix to vector
    w1 = np.ndarray.flatten(w1)
    w2 = np.ndarray.flatten(w2)

    # merge to one vector
    w = np.concatenate((w1, w2))

    return w


# to be used when batch size = 1
def gradients_check(params, x, y_batched, activation_func, index, net_gradients):

    h = 1e-4

    # Iterate over all indexes in x
    for idx, param in enumerate(params):
        # This will be our numerical gradient
        ng = np.zeros(param.shape)
        for j in range(ng.shape[0]):
            for k in range(ng.shape[1]):
                param_plus = np.copy(param)  # create a copy with different reference
                param_minus = np.copy(param)  # create a copy with different reference

                param_plus[j, k] += h
                param_minus[j, k] -= h

                add_params = copy(params)
                min_params = copy(params)
                add_params[idx] = param_plus
                min_params[idx] = param_minus

                y_hat_plus = forward(x, add_params, activation_func)[0]
                y_hat_minus = forward(x, min_params, activation_func)[0]

                cost_plus = float(-np.log(y_hat_plus[int(y_batched[index])]))  # accumulate batch loss
                cost_minus = float(-np.log(y_hat_minus[int(y_batched[index])]))  # accumulate batch loss

                numeric_gradient = (cost_plus - cost_minus) / (2.0 * h)

                reldiff = abs(numeric_gradient - net_gradients[idx][j, k]) / max(1, abs(numeric_gradient), abs(net_gradients[idx][j, k]))

                if reldiff > 1e-5:
                    print "Gradient check failed."
                    print "my gradient: %f \t Numerical gradient: %f" % (net_gradients[idx][j, k], numeric_gradient)
                    return


# calc accuracy on dataset
def accuracy_on_dataset(data_x, data_y, params, activation_func):
    good = bad = 0.0
    for index, x in enumerate(data_x):
        pred = np.argmax(forward(x, params, activation_func)[0])
        if pred == int(data_y[index]):
            good += 1
        else:
            bad += 1
        pass

    return good / (good + bad)


# train model
def train(train_x, train_y, val_x, val_y, params, epochs, learning_rate, reg_lambda, activation_func, batch_size, check_gradients=False):

    num_examples = np.shape(train_x)[0]
    best_dev_accuracy = 0.0

    # train model
    for epoch in range(epochs):
        # shuffle
        train_x, train_y = randomize(train_x, train_y)

        # split the data by batch size
        train_x_batched = np.array_split(train_x, max(1.0, math.floor(num_examples/batch_size)))
        train_y_batched = np.array_split(train_y, max(1.0, math.floor(num_examples/batch_size)))

        num_batches = float(len(train_x_batched))
        cumm_loss = 0.0

        # train
        for batch_ind, curr_batch in enumerate(train_x_batched):
            # get batch targets
            y_batched = train_y_batched[batch_ind]
            batch_loss = dL_dw1_acc =  dL_db1_acc = dL_dw2_acc = dL_db2_acc = 0.0
            for index, example in enumerate(curr_batch):
                y_hat, h1 = forward(example, params, activation_func)  # get prediction and hidden layer activation
                batch_loss += float(-np.log(y_hat[int(y_batched[index])]))  # accumulate batch loss

                # convert y to 1-hot vector
                y_one_hot = np.zeros(np.shape(y_hat))
                y_one_hot[int(y_batched[index])] = 1

                # accumulate gradients
                dL_dw1, dL_db1, dL_dw2, dL_db2 = calc_gradients(y_one_hot, example, y_hat, h1, params, activation_func)
                dL_dw1_acc += dL_dw1
                dL_db1_acc += dL_db1
                dL_dw2_acc += dL_dw2
                dL_db2_acc += dL_db2

                # gradient check
                if check_gradients:
                    gradients = [dL_dw1, dL_db1, dL_dw2, dL_db2]
                    gradients_check(params, example, y_batched, activation_func, index, gradients)

            # calc batch loss with regularization
            w_weights = float(np.dot(weights_to_vec(params).T, weights_to_vec(params)))
            cumm_loss += (1.0/batch_size)*batch_loss
            cumm_loss += 1/2 * reg_lambda * w_weights

            # update gradients:
            params[0] = params[0] - learning_rate * ((1.0 / batch_size) * dL_dw1_acc + reg_lambda * params[0])  # update w1
            params[1] = params[1] - learning_rate * (1.0 / batch_size) * dL_db1_acc  # update b1
            params[2] = params[2] - learning_rate * ((1.0 / batch_size) * dL_dw2_acc + reg_lambda * params[2])  # update w2
            params[3] = params[3] - learning_rate * (1.0 / batch_size) * dL_db2_acc  # update b2

        # get accuracy on train and dev sets
        train_accuracy = accuracy_on_dataset(train_x, train_y, params, activation_func)
        dev_accuracy = accuracy_on_dataset(val_x, val_y, params, activation_func)

        # save best model
        if dev_accuracy > best_dev_accuracy or epoch == 0:
            best_params = np.copy(params)
            best_dev_accuracy = dev_accuracy

        print epoch+1, cumm_loss/num_batches, train_accuracy, dev_accuracy

    return best_params


# learn the task at hand
def run(layers, epochs, learning_rate, reg_lambda, activation_func, batch_size, check_gradients, prediction_file):

    # get train and val sets
    train_x, train_y, val_x, val_y = import_train_data()

    # normalize sets
    train_x = train_x / 255.0
    val_x = val_x / 255.0

    # initialize weights according to Glorot initialization
    params = intialize_weights(layers)
    params = train(train_x, train_y, val_x, val_y, params, epochs, learning_rate, reg_lambda, activation_func, batch_size, check_gradients)

    # predict on test set
    test_x = np.loadtxt("test_x") / 255.0
    csv = open(prediction_file, "w")
    for index, x in enumerate(test_x):
        pred = int(np.argmax(forward(x, params, activation_func)[0]))
        csv.write(str(pred) + "\n")


def plot_test_images():
    import matplotlib.pyplot as plt
    test_x = np.loadtxt("test_x")
    for i in range(1000):
        t = test_x[i].reshape(28, 28)
        plt.imsave("./test_pics/" + str(i+1)+".png", t, cmap='Greys')


if __name__ == '__main__':

    # configurations:
    layers = [784, 150, 10]
    epochs = 25
    learning_rate = 0.002
    reg_lambda = 0.0
    activation_func = "relu" # supports: sigmoid, tanh, ReLU
    batch_size = 1.0
    check_gradients = False
    prediction_file = "./test.pred"

    run(layers, epochs, learning_rate, reg_lambda, activation_func, batch_size, check_gradients, prediction_file)