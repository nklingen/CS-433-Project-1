import numpy as np
from proj1_helpers import *
DATA_TRAIN_PATH = '../data/train.csv'

def sigmoid(t):
    """apply sigmoid function on t."""
    return (np.exp(t)/(1 + np.exp(t)))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    s = sigmoid(tx @ w)
    # print(np.shape(s), np.shape(y))
    a = y.T @ np.log(s)
    b = (1 - y).T @ np.log(1 - s)
    return (-1) * (a + b)

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return np.dot(tx.T, sigmoid(tx @ w) - y)

def logistic_regression(y, tx, w):
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    hessian = calculate_hessian(y, tx,w)

    return loss, gradient, hessian

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    # compute the cost: TODO
    cost = calculate_loss(y, tx, w)

    # compute the gradient: TODO
    grad = calculate_gradient(y, tx, w)

    # update w: TODO
    w = w - gamma * grad
    return cost, w

def logistic_regression_gradient_descent_demo(y, x):
    # init parameters
    max_iter = 100
    threshold = 1e-8
    gamma = 0.01
    losses = []

    print(x.shape)
    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_gradient_descent")
    print("loss={l}".format(l=calculate_loss(y, tx, w)))
    print("w = ", w)


def main():
    ratio = 0.02
    seed = 3
    ## we noticed that the first model had the best accuracy, without any pre-processing.
    ## we then applied ridge regression to the first model.
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    x_tr, x_te, y_tr, y_te = split_data(tX, y, ratio, seed)
    logistic_regression_gradient_descent_demo(y_tr, x_tr)

if __name__== "__main__":
    main()
