import numpy as np
from proj1_helpers import *


################################################################################
'''-------------------------- Least squares GD ------------------------------'''
def compute_gradient(y, tx, w):
    N = y.shape[0]
    gradient = (-1/N)*tx.T@(y-tx@w)
    return gradient

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)

        # update w by gradient
        w = w - gamma*gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Step " + str(n_iter + 1) + "/" + str(max_iters) + " loss = " + str(loss))

    return ws[-1], losses[-1]




################################################################################
'''---------------------- Least squares SGD ---------------------------------'''
def compute_loss(y, tx, w):
    MSE = 1/(2*y.shape[0])*np.sum(np.square(y-np.dot(tx,w)))
    return MSE

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # compute gradient and loss
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(minibatch_y, minibatch_tx, w)

            # update w by gradient
            w = w - gamma*gradient

            # store w and loss
            ws.append(w)
            losses.append(loss)

    return ws[len(ws)-1], losses[len(losses)-1]




################################################################################
'''--------------------------- Least squares --------------------------------'''
def least_squares(y, tx):
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    return w




################################################################################
'''------------------------- Ridge regression -------------------------------'''
def ridge_regression(y, tx, lambda_):
    a = (1/len(y))*(tx.T @ tx) + 2 * (lambda_ * np.identity(tx.shape[1]))
    b = (1/len(y))*(tx.T @ y)
    w = np.linalg.solve(a,b)
    MSE = compute_loss(y, tx, w)
    return w, MSE




################################################################################
''' ------------------ Logistic regression using GD -------------------------'''
def sigmoid(t):
    return np.exp(t)/(1+np.exp(t))

def compute_loss_neg_log_likehood(y, tx, w):
    # compute the cost by negative log likelihood
    N = y.shape[0]
    s = sigmoid(np.matmul(tx, w))
    return -np.sum((y*np.log(s) + (1-y)*np.log(1-s))/N)

def gradient_log_reg(y, tx, w):
    # compute the gradient of loss
    s = sigmoid(np.matmul(tx, w))
    return np.matmul(np.transpose(tx), s - y)

def grad_log_reg_iter(y, tx, w, gamma):
    # One iteration: calculate gradient descent using log reg
    grad = gradient_log_reg(y, tx, w)
    w = w - gamma * grad
    return w

def logistic_regression(y, tx, initial_w, max_iters, gamma, batch_size=1):

    w = initial_w.reshape(-1,1)
    y = y.reshape(-1,1)
    loss = 0.0

    for n_iter in range(max_iters):
        # get loss and update w.
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            w = grad_log_reg_iter(minibatch_y, minibatch_tx, w, gamma)

        loss = compute_loss_neg_log_likehood(y, tx, w)
        print("Logistic Regression ({bi}/{ti}): loss={l}".format(bi=n_iter+1, ti=max_iters, l=loss))
    return w, loss



################################################################################
''' --------------- Regularized logistic regression using GD ---------------- '''
def grad_reg_log_reg_iter(y, tx, lambda_, w, gamma):
    loss = compute_loss_neg_log_likehood(y, tx, w) + 0.5*lambda_*np.sum(w*w)
    gradient = gradient_log_reg(y, tx, w) + lambda_*w
    w = w - gamma * gradient
    return loss, w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, batch_size=1, lRdecay=False, lRdecayRate=0.7):
    # Reshape y and w
    y = y.reshape(-1,1)
    w = initial_w.reshape(-1,1)
    threshold = 1e-8 # for converge criterion
    loss = 0.0
    regLossParam = 0.5
    losses = []

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            _, w = grad_reg_log_reg_iter(minibatch_y, minibatch_tx, lambda_, w, gamma)
        loss_neg_likehood = compute_loss_neg_log_likehood(y, tx, w)
        regLoss = regLossParam * lambda_ * np.sum(w * w)
        loss = loss_neg_likehood + regLoss


        print("Reg Log Regression({bi}/{ti}): Loss={l} l1={loss_neg_likehood} l2={regLoss}".format(
            bi=n_iter+1, ti=max_iters, l=loss, loss_neg_likehood=loss_neg_likehood, regLoss=regLoss))

        if n_iter % 100 == 0 and lRdecay:
            gamma = gamma * lRdecayRate

        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, loss
