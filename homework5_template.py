import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 40  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient

MINIBATCH_SIZE = 16


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
def loadData(which):
    images = np.load("fashion_mnist_{}_images.npy".format(which)).T / 255.
    labels = np.load("fashion_mnist_{}_labels.npy".format(which))

    # TODO: Convert labels vector to one-hot matrix (C x N). (See previous HW)
    # ...

    return images, labels


# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss, accuracy,
# as well as the intermediate values of the NN.
def fCE(X, Y, w):
    W1, b1, W2, b2 = unpack(w)

    # TODO: CALCULATE LOSS

    return cost, acc, z1, h1, W1, W2, yhat


# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE(X, Y, w):
    W1, b1, W2, b2 = unpack(w)
    diff = 0
    g = 0

    # TODO: CALCULATE GRADIANT

    return grad


# Given training and testing datasets and an initial set of weights/biases b,
# train the NN.
def train(trainX, trainY, testX, testY, w):

    # TODO: SGD TIME (See previous HW)

    pass


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
    idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    print("Numerical gradient:")
    print(scipy.optimize.approx_fprime(w, lambda w_:
    fCE(np.atleast_2d(trainX[:, idxs]), np.atleast_2d(trainY[:, idxs]), w_)[0], 1e-10))
    print("Analytical gradient:")
    print(gradCE(np.atleast_2d(trainX[:, idxs]), np.atleast_2d(trainY[:, idxs]), w))
    print("Discrepancy:")
    print(
        scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[:, idxs]), np.atleast_2d(trainY[:, idxs]), w_)[0], \
                                  lambda w_: gradCE(np.atleast_2d(trainX[:, idxs]), np.atleast_2d(trainY[:, idxs]), w_), \
                                  w))

    # Train the network using SGD.
    train(trainX, trainY, testX, testY, w)
