import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

NUM_INPUT = 784  # Number of input neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient

NUM_HIDDEN_OPTIONS = [30, 40, 50]
LEARNING_RATE_OPTIONS = [.001, .005, .01, .05, .1, .5]
MINIBATCH_SIZE_OPTIONS = [16, 32, 64, 128, 256]
EPOCH_NUM_OPTIONS = [1, 2, 4, 8, 16, 32, 64, 128]
REGULARIZATION_STRENGTH_OPTIONS = [.05, .1, .5]

NUM_HIDDEN = NUM_HIDDEN_OPTIONS[0]  # Number of hidden neurons [HYPERPARAMETER TUNING VALUE]
LEARNING_RATE = LEARNING_RATE_OPTIONS[0]  # [HYPERPARAMETER TUNING VALUE]
MINIBATCH_SIZE = MINIBATCH_SIZE_OPTIONS[0]  # [HYPERPARAMETER TUNING VALUE]
EPOCH_NUM = EPOCH_NUM_OPTIONS[0]  # [HYPERPARAMETER TUNING VALUE]
REGULARIZATION_STRENGTH = REGULARIZATION_STRENGTH_OPTIONS[0]  # [HYPERPARAMETER TUNING VALUE]


# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.
def unpack(w):
    # Unpack arguments
    start = 0
    end = NUM_HIDDEN * NUM_INPUT
    W1 = w[0:end]
    start = end
    end = end + NUM_HIDDEN
    b1 = w[start:end]
    start = end
    end = end + NUM_OUTPUT * NUM_HIDDEN
    W2 = w[start:end]
    start = end
    end = end + NUM_OUTPUT
    b2 = w[start:end]
    # Convert from vectors into matrices
    W1 = W1.reshape(NUM_HIDDEN, NUM_INPUT)
    W2 = W2.reshape(NUM_OUTPUT, NUM_HIDDEN)
    return W1, b1, W2, b2


# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
# This is useful for performing a gradient check with check_grad.
def pack(W1, b1, W2, b2):
    return np.hstack((W1.flatten(), b1, W2.flatten(), b2))


# Load the images and labels from a specified dataset (train or test).
# Return images and one hot lables
def loadData(which):
    images = np.load("fashion_mnist_{}_images.npy".format(which)) / 255.
    labels = np.load("fashion_mnist_{}_labels.npy".format(which))
    sample_num, data_len = np.shape(images)
    lable_len = 10
    labels_OH = np.zeros((sample_num, lable_len))

    for i in range(lable_len):
        for j in range(sample_num):
            if labels[j] == i:
                labels_OH[j][i] = 1

    return images, labels_OH


# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss, accuracy,
# as well as the intermediate values of the NN.
def fCE(X, Y, w):
    W1, b1, W2, b2 = unpack(w)

    sample_num, data_len = X.shape
    sample_num_y, class_len = Y.shape

    bias_vector = np.atleast_2d(np.ones(sample_num)).T
    X = np.hstack((X, bias_vector))

    n = X.shape[1]

    print("n= ", n)
    print("y shape= ", Y.shape)
    z1 = np.dot(np.hstack((W1, np.atleast_2d(np.ones(W1.shape[0])).T)), X.T)
    #z1 = np.vstack((z1.T, b1.T))
    myList = []
    for sublist in z1:
        thisRow = []
        for element in sublist:
            thisRow.append(relu(element))
        myList.append(thisRow)

    h1 = np.array(myList)
    z2 = np.dot(W2, h1)
    #z2 = np.vstack((z2.T, b2.T))
    yhat = np.exp(z2) / np.sum(np.exp(z2), axis=None)

    smallSum = np.sum(np.dot(Y, np.log(yhat[0:n])), axis=1)
    bigSum = np.sum(smallSum, axis=0)
    loss = (-1 / n) * bigSum
    acc = -1

    cost = loss
    return cost, acc, z1, h1, W1, W2, yhat[0:n]  # deciding whether or not to "clip" off the bias on yhat (see the [0 to n] )


# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE(X, Y, w):
    W1, b1, W2, b2 = unpack(w)

    cost, acc, z1, h1, W1, W2, yhat = fCE(X, Y, w)
    print("Y shape = ", Y.shape)
    print("y^hat shape = ", yhat.shape)
    deltaB2 = (yhat - Y.T)
    deltaW2 = np.dot(deltaB2,h1.T)
    deltaB1 = np.multiply(np.dot(deltaB2.T,W2), (reluPrime(z1.T))).T
    deltaW1 = deltaB1 * X.T

    return pack(deltaW1, deltaB1, deltaW2, deltaB2)


def relu(z):
    return max(0.0, z)


def reluPrime(z):
    if (z > 0):
        return 1
    else:
        return 0


# Given training and testing datasets and an initial set of weights/biases b,
# train the NN.
def train(trainX, trainY, w):
    sample_num, data_len = trainX.shape
    sample_num_y, class_len = trainY.shape

    # graph code
    # out = []
    # file_name = 'HW5_part1.csv'

    for i in range(EPOCH_NUM):
        random_inds = np.arange(sample_num)
        np.random.shuffle(random_inds)
        index_index = 0
        print_i = 0
        for j in range(int(sample_num / MINIBATCH_SIZE) - 1):
            # select minibatch
            batch = np.empty((MINIBATCH_SIZE, data_len))
            batch_lables = np.empty((MINIBATCH_SIZE, class_len))

            for k in range(MINIBATCH_SIZE):  # check for np optimization
                if index_index < sample_num:
                    ind = random_inds[index_index]
                    batch[k] = trainX[ind]
                    batch_lables[k] = trainY[ind]
                    index_index += 1

            gradient = gradCE(trainX, trainY, w)
            w = w - LEARNING_RATE * gradient  # make element wise

        cost, acc, z1, h1, W1, W2, yhat = fCE(trainX, trainY, w)
        print("Cross-entropy loss: ", cost, "PCC: ", acc)
        # out.append(f)
        # if (i % 50 == 0):
        # df = pd.DataFrame(out, columns=["fMSE"])
        # df.to_csv(file_name)

    return w


def findBestHyperparaneters(trainX, trainY, w):
    data_len, sample_num = trainX.shape
    data_len_y, class_len = trainY.shape

    NUM_HIDDEN_OPTIONS_len = len(NUM_HIDDEN_OPTIONS)
    LEARNING_RATE_OPTIONS_len = len(LEARNING_RATE_OPTIONS)
    MINIBATCH_SIZE_OPTIONS_len = len(MINIBATCH_SIZE_OPTIONS)
    EPOCH_NUM_OPTIONS_len = len(EPOCH_NUM_OPTIONS)
    REGULARIZATION_STRENGTH_OPTIONS_len = len(REGULARIZATION_STRENGTH_OPTIONS)

    total_len = (
            NUM_HIDDEN_OPTIONS_len * LEARNING_RATE_OPTIONS_len * MINIBATCH_SIZE_OPTIONS_len * EPOCH_NUM_OPTIONS_len * REGULARIZATION_STRENGTH_OPTIONS_len)
    print("Number of total iterations: ", total_len)
    start_time = time.time()
    best_w = w
    best_loss = 1
    best_NUM_HIDDEN = 0
    best_LEARNING_RATE = 0
    best_MINIBATCH_SIZE = 0
    best_EPOCH_NUM = 0
    best_REGULARIZATION_STRENGTH = 0

    # TODO get validation data to test with

    for a in range(NUM_HIDDEN_OPTIONS_len):
        NUM_HIDDEN = NUM_HIDDEN_OPTIONS[a]
        for b in range(LEARNING_RATE_OPTIONS_len):
            LEARNING_RATE = LEARNING_RATE_OPTIONS[b]
            for c in range(MINIBATCH_SIZE_OPTIONS_len):
                MINIBATCH_SIZE = MINIBATCH_SIZE_OPTIONS[c]
                for d in range(EPOCH_NUM_OPTIONS_len):
                    EPOCH_NUM = EPOCH_NUM_OPTIONS[d]
                    for e in range(REGULARIZATION_STRENGTH_OPTIONS_len):
                        REGULARIZATION_STRENGTH = REGULARIZATION_STRENGTH_OPTIONS[e]
                        w = train(trainX, trainY, w)
                        loss = fCE(trainX, trainY, w)[0]
                        if (loss < best_loss):
                            best_loss = loss
                            best_w = w
                            best_NUM_HIDDEN = NUM_HIDDEN
                            best_LEARNING_RATE = LEARNING_RATE
                            best_MINIBATCH_SIZE = MINIBATCH_SIZE
                            best_EPOCH_NUM = EPOCH_NUM
                            best_REGULARIZATION_STRENGTH = REGULARIZATION_STRENGTH
                            print("New best hyperparameters with loss of: ", best_loss, ". NUM_HIDDEN: ", NUM_HIDDEN,
                                  " LEARNING_RATE: ", LEARNING_RATE, " MINIBATCH_SIZE: ", MINIBATCH_SIZE,
                                  " EPOCH_NUM: ", EPOCH_NUM, " REGULARIZATION_STRENGTH: ", REGULARIZATION_STRENGTH)
    print("Globals are now set to best hyperameter values out of available options")

    NUM_HIDDEN = best_NUM_HIDDEN
    LEARNING_RATE = best_LEARNING_RATE
    MINIBATCH_SIZE = best_MINIBATCH_SIZE
    EPOCH_NUM = best_EPOCH_NUM
    REGULARIZATION_STRENGTH = best_REGULARIZATION_STRENGTH

    return best_w


if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY = loadData("train")
        testX, testY = loadData("test")

    # Initialize weights randomly
    W1 = 2 * (np.random.random(size=(NUM_HIDDEN, NUM_INPUT)) / NUM_INPUT ** 0.5) - 1. / NUM_INPUT ** 0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2 * (np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN)) / NUM_HIDDEN ** 0.5) - 1. / NUM_HIDDEN ** 0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)

    # Concatenate all the weights and biases into one vector; this is necessary for check_grad
    w = pack(W1, b1, W2, b2)

    # Check that the gradient is correct on just a few examples (randomly drawn).
    idxs = np.random.permutation((trainX.T).shape[0])[0:NUM_CHECK]

    code_test_x = np.atleast_2d(trainX[idxs])
    code_test_y = np.atleast_2d(trainY[idxs])

    anal_grad = gradCE(code_test_x, code_test_y, w)
    print(anal_grad)

    testMyFCE = fCE(np.atleast_2d(trainX[idxs]), np.atleast_2d(trainY[idxs]), w)[0]
    print(testMyFCE)

    print("(main) Y shape = ", trainY.shape)

    print("Numerical gradient:")
    print(scipy.optimize.approx_fprime(w, lambda w_:
    fCE(code_test_x, np.atleast_2d(trainY[idxs]), w_)[1], 1e-10))
    print("Analytical gradient:")
    print(anal_grad)
    print("Discrepancy:")
    print(
        scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[:, idxs]), np.atleast_2d(trainY[idxs]), w_)[0], \
                                  lambda w_: gradCE(np.atleast_2d(trainX[:, idxs]), np.atleast_2d(trainY[idxs]), w_), \
                                  w))

    # Train the network using SGD.
    train(trainX, trainY, testX, testY, w)
