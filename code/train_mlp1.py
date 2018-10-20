import mlp1 as ml
import random

STUDENT={'name': 'Idan Achituve',
         'ID': '300083029'}


def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.

    import utils as ut
    import numpy as np

    count_vec = []  # a list of counts with length = in dimention
    for i in range(len(ut.vocab)):
        count_vec.append(0)

    for en_bi in features:
        count_vec[en_bi] += 1

    return np.array(count_vec)


def accuracy_on_dataset(dataset, params, Xor = False):
    good = bad = 0.0
    for label, features in dataset:

        if Xor == False:
            x = feats_to_vec(features)  # convert features to a vector
        else:
            x = np.array(features)  # convert features to a vector

        pred = ml.predict(x, params)
        if pred == label:
            good += 1
        else:
            bad += 1
        pass

    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params, reg_lambda=None, dropout = False ,Xor = False):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """

    for I in xrange(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:

            import numpy as np

            if Xor == False:
                x = feats_to_vec(features)  # convert features to a vector.
            else:
                x = np.array(features)  # convert features to a vector.
            y = label  # convert the label to number if needed.

            if dropout == True:
                ber = np.random.choice([0, 1], size=params[1].shape, p=[1. / 2, 1. / 2])  # random bernoli vector for dropout at the size of the hidden layer
                loss, grads = ml.loss_and_gradients(x, y, params, reg_lambda, ber)
            else:
                loss, grads = ml.loss_and_gradients(x, y, params, reg_lambda, None)

            cum_loss += loss

            w_hid = grads[0] * learning_rate
            b_hid = grads[1] * learning_rate
            w_out = grads[2] * learning_rate
            b_out = grads[3] * learning_rate

            params[0] = np.subtract(params[0], w_hid)
            params[1] = np.subtract(params[1], b_hid)
            params[2] = np.subtract(params[2], w_out)
            params[3] = np.subtract(params[3], b_out)

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params,Xor)
        if Xor==False:
            dev_accuracy = accuracy_on_dataset(dev_data, params)
            print I, train_loss, train_accuracy, dev_accuracy
        else:
            print I, train_loss, train_accuracy
    return params

if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    # ...
    import utils as ut
    import numpy as np
    import math

    Xor = False # run on Xor data or text data


    if Xor == False:
        in_dim = len(ut.vocab)  # the number of nurons in the input layer = 600 (vocabulary size)
        out_dim = len(ut.L2I)  # the number of nurons in the output layer as the number of labels

        # This part convert the original data read from files to a tuple of (enumerated label, [enumerated baigrams])
        train_data = []
        for l, baigrams in ut.TRAIN:
            label = ut.L2I.get(l)  # replace the label with the enumeration
            vocab_biagrams = []  # maintain list for baigrams from the current example that are in the vocabulary
            for baigram in baigrams:
                if baigram in ut.vocab:
                    vocab_biagrams.append(ut.F2I.get(baigram))
            train_data.append((label, vocab_biagrams))

        dev_data = []
        for l, baigrams in ut.DEV:
            label = ut.L2I.get(l)  # replace the label with the enumeration
            vocab_biagrams = []  # list for baigrams from the current example that are in the vocabulary
            for baigram in baigrams:
                if baigram in ut.vocab:  # the advantage is 2-fold: extract most relevant features, make sure that in baigrams appeared in validation set that did not exist in training set enter the
                    vocab_biagrams.append(ut.F2I.get(baigram))
                dev_data.append((label, vocab_biagrams))


        # configurable:
        num_iterations = 30
        learning_rate = 0.002
        hid_dim = 10
        reg_lambda = 0

        # initialization according to Xavier suggestion
        n = float(len(train_data)) # number of examples
        m = float(in_dim) # input size
        epsilon = math.sqrt(6.0) / math.sqrt(m + n) # uniform range

        params = ml.create_classifier(in_dim, hid_dim,out_dim)

        params_rnd = []
        for i in range(len(params)):
            if i%2==0: # Weights parameters
                W_rand = []
                for j in range(0, params[i].shape[0]):
                    row = np.random.uniform(-epsilon, epsilon, params[i].shape[1])
                    W_rand.append(row)
                params_rnd.append(np.array(W_rand))
            else:
                b_rnd = np.random.uniform(-epsilon, epsilon, params[i].shape[0])
                params_rnd.append(np.array(b_rnd))

        # traing model:
        trained_params = train_classifier(train_data, dev_data, num_iterations,learning_rate, params_rnd)

    else:
        import xor_data as xd
        in_dim = 2
        out_dim = 2

        # configurable:
        num_iterations = 30
        learning_rate = 0.3
        hid_dim = 10
        reg_lambda = 0.0

        params = ml.create_classifier(in_dim, hid_dim, out_dim)

        # initialization according to Xavier suggestion
        n = float(len(xd.data))  # number of examples
        m = float(in_dim)  # input size
        epsilon = math.sqrt(6.0) / math.sqrt(m + n)  # uniform range

        params = ml.create_classifier(in_dim, hid_dim, out_dim)

        params_rnd = []
        for i in range(len(params)):
            if i % 2 == 0:  # Weights parameters
                W_rand = []
                for j in range(0, params[i].shape[0]):
                    row = np.random.uniform(-epsilon, epsilon, params[i].shape[1])
                    W_rand.append(row)
                params_rnd.append(np.array(W_rand))
            else:
                b_rnd = np.random.uniform(-epsilon, epsilon, params[i].shape[0])
                params_rnd.append(np.array(b_rnd))

        trained_params = train_classifier(xd.data, xd.data,num_iterations,learning_rate, params_rnd,None,None,True)