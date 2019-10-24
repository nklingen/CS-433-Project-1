# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""

    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

"Constructs the X-tilda array with the bias variable"
def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


"Used to standardize height with with a mean 0 and std 1"
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
     # set seed to have reproducible/consistent results

    # create np array of random indices
    num_row = len(y)
    indices = np.random.permutation(len(y))

    # split the random indice array into training / test
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]

    # constructs array for training / test given the chosen incides.
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te

def plot_fitted_curve(y, x, weights, degree, ax):
    """plot the fitted curve."""
    ax.scatter(x, y, color='b', s=12, facecolors='none', edgecolors='r')
    xvals = np.arange(min(x) - 0.1, max(x) + 0.1, 0.1)
    tx = build_poly(xvals, degree)
    f = tx.dot(weights)
    ax.plot(xvals, f)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Polynomial degree " + str(degree))


def replace_nan_with_mean(tX, nanValue = -999):
    ''' This function replaces all the nanValues in the columns with the column'''
    ''' mean (this mean is calculated by considering all the other lines that  '''
    ''' do not contain nanValue)'''

    isNA = (tX == nanValue)
    # Compute for each column the sum of the elements != nan
    columnSums = np.sum(tX * (1 - isNA), axis = 0, keepdims = True)

    # Compute for each column the number of values != nan
    nr_column = np.sum(1 - isNA, axis = 0, keepdims = True)

    #Compute the mean for each column, for column i, the mean is at mean_cols[i]
    mean_cols = columnSums / nr_column

    # Replace all the nanValues with the corresponding mean of the column
    tX = isNA * mean_cols + (1 - isNA) * tX
    return tX


def replace_with_constant(tX, constant, nanValue = -999):
    ''' This function replaces all the nanValues in the column i with the constant[i]'''
    isNA = (tX == nanValue)
    tX = isNA * constant + (1 - isNA)*tX
    return tX

def build_poly(x, degree):
    poly = x
    for deg in range(2, degree+1):
        poly = np.concatenate((poly, np.power(x, deg)), axis = 1)
    return poly


def compute_loss(y, tx, w):
    
    MSE = 1/(2*y.shape[0])*np.sum(np.square(y-np.dot(tx,w)))
    
    return MSE

def sigmoid(t):
    """apply sigmoid function on t."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    sigmoid = np.exp(t)/(1+np.exp(t))
    return sigmoid
    # ***************************************************
    raise NotImplementedError
    
def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    loss = np.sum(np.log(1+np.exp(tx@w))) - np.sum(y*tx@w)
    return loss
    # ***************************************************
    raise NotImplementedError
    
def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    gradient = np.dot(tx.T, sigmoid(np.dot(tx,w))-y)
    return gradient
    # ***************************************************
    raise NotImplementedError

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate hessian: TODO
    S = sigmoid(tx@w) @ (1-sigmoid(tx@w)).T
    Hessian = tx.T @ S @ tx
    return Hessian

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    # return loss, gradient, and hessian: TODO
    loss = calculate_loss(y, tx, w) + (lambda_/2) * np.linalg.norm(w, ord=2) ** 2
    gradient = calculate_gradient(y, tx, w) + lambda_ * w 
    hessian = calculate_hessian(y, tx, w) + 2 * np.diag(np.ones(len(w)))
    return loss, gradient, hessian

    
def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    # ***************************************************
    # return loss, gradient
    loss, gradient, hessian = penalized_logistic_regression(y, tx, w, lambda_)
    # ***************************************************
    # update w
    w = w - gamma * gradient
    
    return loss, w
