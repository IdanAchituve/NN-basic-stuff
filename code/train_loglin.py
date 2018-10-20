import loglinear as ll
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

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)

        import loglinear as ll

        x = feats_to_vec(features) # convert features to a vector

        pred = ll.predict(x, params)
        if pred == label:
            good += 1
        else:
            bad += 1
        pass

    return good / (good + bad)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """

    for I in xrange(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = label                  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x,y,params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.

            import numpy as np

            w_up= grads[0] * learning_rate
            b_up= grads[1] * learning_rate

            params[0] = np.subtract(params[0], w_up)
            params[1] = np.subtract(params[1], b_up)



        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    
    # ...
    import utils as ut

    in_dim = len(ut.vocab) # the number of nurons in the input layer = 600 (vocabulary size)
    out_dim = len(ut.L2I) # the number of nurons in the output layer as the number of labels

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
    num_iterations = 40
    learning_rate = 0.002

    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

