import time
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize

NUM_INPUT = 784  # Number of input neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient

GLOBAL_DEBUG = False

cost = 0.0
acc = 0.0

NUM_HIDDEN_OPTIONS = [30, 40, 50]
LEARNING_RATE_OPTIONS = [.001, .005, .01, .05, .1, .5]
MINIBATCH_SIZE_OPTIONS = [16, 32, 64, 128, 256]
EPOCH_NUM_OPTIONS = [1, 2, 4, 8, 16, 32, 64]
REGULARIZATION_STRENGTH_OPTIONS = [.05, .1, .5]

NUM_HIDDEN = NUM_HIDDEN_OPTIONS[0]  # Number of hidden neurons [HYPERPARAMETER TUNING VALUE]
LEARNING_RATE = LEARNING_RATE_OPTIONS[0]  # [HYPERPARAMETER TUNING VALUE]
MINIBATCH_SIZE = MINIBATCH_SIZE_OPTIONS[0]  # [HYPERPARAMETER TUNING VALUE]
EPOCH_NUM = EPOCH_NUM_OPTIONS[5]  # [HYPERPARAMETER TUNING VALUE]
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

    n = X.shape[0]
    relu_vec = np.vectorize(relu)

    z1 = np.dot(np.hstack((W1, np.atleast_2d(np.ones(W1.shape[0])).T)), X.T)
    h1 = relu_vec(z1)
    z2 = np.dot(W2, h1)
    # z2 = np.vstack((z2.T, b2.T))
    yhat = np.exp(z2) / np.sum(np.exp(z2), axis=0)

    bug = np.multiply(Y.T, np.log(yhat))
    smallSum = np.sum(bug, axis=0)
    bigSum = np.sum(smallSum, axis=0)
    #alpha = 0.1
    alpha = REGULARIZATION_STRENGTH
    loss = (-1 / n) * bigSum  + ((alpha/(2*n))*np.dot(w.T,w))
    global acc, cost
    acc = fPC(Y, yhat.T)

    cost = loss
    if GLOBAL_DEBUG:
        print("PC rate: ", acc)
        print("loss: ", cost)
    return cost, acc, z1, h1, W1, W2, yhat  # deciding whether or not to "clip" off the bias on yhat (see the [0 to n] )


# takes 10 x n one-hot vectors for y and yhat
def fPC(y, yhat):
    n = y.shape[0]
    y_maxes = np.argmax(y, axis=1)
    yhat_maxes = np.argmax(yhat, axis=1)
    pc = np.count_nonzero(y_maxes == yhat_maxes) / n
    return pc


# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE(X, Y, w):
    W1, b1, W2, b2 = unpack(w)

    cost, acc, z1, h1, W1, W2, yhat = fCE(X, Y, w)
    # print("Y shape = ", Y.shape)
    # print("y^hat shape = ", yhat.shape)
    diff = (yhat - Y.T)
    deltaB2 = np.sum(diff, axis=1)
    deltaW2 = np.dot(diff, h1.T)
    myList = []
    for sublist in z1:
        thisRow = []
        for element in sublist:
            thisRow.append(reluPrime(element))
        myList.append(thisRow)

    helper = np.array(myList)
    g = np.multiply(np.dot(diff.T, W2), helper.T).T
    deltaB1 = np.sum(g, axis=1)
    deltaW1 = np.dot(g, X)
    ##deltaB1 = np.multiply(np.dot(deltaB2.T, W2), helper.T).T
    # deltaW1 = np.dot(deltaB1, X)

    return pack(deltaW1, deltaB1, deltaW2, deltaB2)


relu = lambda z: max(0.0, z)


# def relu(z):
#     return max(0.0, z)


def reluPrime(z):
    if (z > 0):
        return 1
    else:
        return 0


def calc_yhat(X, Y, w):
    W1, b1, W2, b2 = unpack(w)
    n = X.shape[1]

    a = np.dot(W2, W1)
    b = np.dot(W2, b1)
    yhat = np.multiply(np.dot(a, X.T).T, (b + b2))

    smallSum = np.dot(Y, np.log(yhat).T)
    bigSum = np.sum(smallSum, axis=0)
    loss = (-1 / n) * bigSum
    acc = -1  # TODO calculate

    cost = loss

    return cost, acc, yhat


# Given training and testing datasets and an initial set of weights/biases b,
# train the NN.
def train(trainX, trainY, w):
    sample_num, data_len = trainX.shape
    sample_num_y, class_len = trainY.shape
    file_name = 'HW5_plot_bus.csv'
    os.remove(file_name)
    # graph code
    # out = []
    # file_name = 'HW5_part1.csv'
    global cost, acc
    descent_step = 0

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

            gradient = gradCE(batch, batch_lables, w)
            w = w - LEARNING_RATE * gradient
            global cost, acc

            file_name = 'HW5_plot_bus.csv'
            cost_output = cost
            acc_output = acc
            df = pd.DataFrame([[descent_step, cost_output, acc_output]])
            df.to_csv(file_name, mode='a', header=False, index=False)
            descent_step += 1

        # cost, acc, yhat = calc_yhat(trainX, trainY, w)
        print("Epoch: ", i, "Cross-entropy loss: ", cost, "PCC: ", acc)
    return w


def findBestHyperparaneters(trainX, trainY, w):
    data_len, sample_num = trainX.shape
    data_len_y, class_len = trainY.shape

    NUM_HIDDEN_OPTIONS_len = len(NUM_HIDDEN_OPTIONS)
    LEARNING_RATE_OPTIONS_len = len(LEARNING_RATE_OPTIONS)
    MINIBATCH_SIZE_OPTIONS_len = len(MINIBATCH_SIZE_OPTIONS)
    EPOCH_NUM_OPTIONS_len = len(EPOCH_NUM_OPTIONS)
    REGULARIZATION_STRENGTH_OPTIONS_len = len(REGULARIZATION_STRENGTH_OPTIONS)

    print("---- FINDING OPTIMAL HYPERPARAMETER VALUES.  ----------")
    total_len = (NUM_HIDDEN_OPTIONS_len * LEARNING_RATE_OPTIONS_len * MINIBATCH_SIZE_OPTIONS_len * EPOCH_NUM_OPTIONS_len * REGULARIZATION_STRENGTH_OPTIONS_len)
    print("Number of total iterations: ", total_len)
    iterationCounter = 0
    start_time = time.time()
    best_w = w
    best_cost = 1000000000
    best_NUM_HIDDEN = 0
    best_LEARNING_RATE = 0
    best_MINIBATCH_SIZE = 0
    best_EPOCH_NUM = 0
    best_REGULARIZATION_STRENGTH = 0
    global cost,acc
    # TODO get validation data to test with
    validation_len = int(0.2 * data_len)
    # idxs = np.random.permutation((trainX.T).shape[0])[0:NUM_CHECK]
    validation_idx = np.random.permutation(trainX.T.shape[0])[0:validation_len]
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
                        w = train(trainX[validation_idx], trainY[validation_idx], w)
                        validation_loss = fCE(trainX[validation_idx], trainY[validation_idx], w)[0]

                        print("Iteration ", iterationCounter)
                        iterationCounter = iterationCounter+1;
                        if (validation_loss < best_cost):
                            best_cost = validation_loss
                            best_w = w
                            best_NUM_HIDDEN = NUM_HIDDEN
                            best_LEARNING_RATE = LEARNING_RATE
                            best_MINIBATCH_SIZE = MINIBATCH_SIZE
                            best_EPOCH_NUM = EPOCH_NUM
                            best_REGULARIZATION_STRENGTH = REGULARIZATION_STRENGTH
                            print("New best hyperparameters with loss of: ", best_cost, ". NUM_HIDDEN: ", NUM_HIDDEN,
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

    print("Numerical gradient:")
    num_grad = scipy.optimize.approx_fprime(w, lambda w_:fCE(code_test_x, np.atleast_2d(trainY[idxs]), w_)[0], 1e-10)
    print(num_grad.shape)
    print(num_grad)
    print("Analytical gradient:")
    anal_grad = gradCE(code_test_x, code_test_y, w)
    print(anal_grad.shape)
    print(anal_grad)
    print("Discrepancy:")
    print(
        scipy.optimize.check_grad(lambda w_: fCE(code_test_x, code_test_y, w_)[0],
                                  lambda w_: gradCE(code_test_x, code_test_y, w_),
                                  w))

    # w = train(trainX, trainY, w)
    findBestHyperparaneters(trainX, trainY, w)
    print("Now training using best hyperparameters.")
    w = train(trainX, trainY, w)

    # Train the network using SGD.
    # train(trainX, trainY, w)
