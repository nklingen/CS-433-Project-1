# This file runs all the required method and print the prediction accuracy


import numpy as np
from helpers import *
from implementations import *

# This function standardizes the data (Reference: Lab2)
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x



def run_least_squares_GD(x_train, y_train, x_validation, y_validation):
    print("Running least squares GD...")
    gamma = 0.08
    max_iters = 300
    initial_w = np.zeros(len(x_train[0]))

    w, loss = least_squares_GD(y_train, x_train, initial_w, max_iters, gamma)

    y_pred = predict_labels(w, x_validation)
    y_acc = np.mean(y_validation == y_pred)
    print("Prediction accuracy = ", y_acc*100, "%")



def run_least_squares_SGD(x_train, y_train, x_validation, y_validation):
    print("\nRunning least squares SGD...")
    gamma = 1e-8
    max_iters = 200
    initial_w = np.zeros(len(x_train[0]))

    w, loss = least_squares_SGD(y_train, x_train, initial_w, max_iters, gamma, 1)

    y_pred = predict_labels(w, x_validation)
    y_acc = np.mean(y_validation == y_pred)
    print("Prediction accuracy = ", y_acc*100, "%")



def run_least_squares(x_train, y_train, x_validation, y_validation):
    print("\nRunning least squares...")

    w = least_squares(y_train, x_train)

    y_pred = predict_labels(w, x_validation)
    y_acc = np.mean(y_validation == y_pred)
    print("Prediction accuracy = ", y_acc*100, "%")



def run_ridge_regression(x_train, y_train, x_validation, y_validation):
    print("\nRunning ridge regression...")

    lambda_ = 0.007
    w, loss = ridge_regression(y_train, x_train, lambda_)

    y_pred = predict_labels(w, x_validation)
    y_acc = np.mean(y_validation == y_pred)
    print("Prediction accuracy = ", y_acc*100, "%")



def run_log_regression(x_train, y_train, x_validation, y_validation):
    print("\nRunning logistic regression...")

    initial_w = np.zeros(len(x_train[0]))
    max_iters = 1000
    gamma = 1e-4

    y_train = y_train > 0

    w, mse = logistic_regression(y_train, x_train, initial_w, max_iters, gamma)

    y_pred = predict_labels(w, x_validation)
    y_acc = np.mean(y_validation.reshape(-1,1)==y_pred)
    print("Prediction accuracy = ", y_acc * 100, "%")



def run_reg_log_regression(x_train, y_train, x_validation, y_validation):
    print("\nRunning reg logistic regression...")

    max_iters = 1000
    gamma = 5e-5
    initial_w = np.zeros(len(x_train[0]))
    lambda_ = 1
    y_train = y_train > 0

    w, loss = reg_logistic_regression(y_train, x_train, lambda_, initial_w, max_iters, gamma, 32)

    y_pred = predict_labels(w, x_validation)
    y_acc = np.mean(y_validation.reshape(-1,1)==y_pred)
    print("Prediction accuracy = ", y_acc*100, "%")

def main():
    # Load the training data
    print("Loading training data...")
    DATA_TRAIN_PATH = 'train.csv'
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    # standardize data
    tX,_,_ = standardize(tX)

    # Split training data
    print("Splitting training data with ratio 0.8 ...\n")
    ratio = 0.8
    seed = 1

    x_train, x_validation, y_train, y_validation = split_data(tX, y, ratio, seed)

    run_least_squares_GD(x_train, y_train, x_validation, y_validation)
    run_least_squares_SGD(x_train, y_train, x_validation, y_validation)
    run_least_squares(x_train, y_train, x_validation, y_validation)
    run_ridge_regression(x_train, y_train, x_validation, y_validation)
    run_log_regression(x_train, y_train, x_validation, y_validation)
    run_reg_log_regression(x_train, y_train, x_validation, y_validation)


if __name__ == "__main__":
    main()
