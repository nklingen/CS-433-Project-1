# Authors: 
# Daniel-Florin Dosaru
# Natasha Ørregaard Klingenbrunn
# Sena Necla Cetin

import numpy as np
from helpers import *
from implementations import *

# Load the training data
print("Loading training data...")
DATA_TRAIN_PATH = 'train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

# Split training data 
print("Splitting training data with ratio 0.8 ...\n")
ratio = 0.8
seed = 1
x_train, x_validation, y_train, y_validation = split_data(tX, y, ratio, seed)

# Running ridge regression
degrees = [1, 2, 3, 4, 5, 6, 7] # the degrees we are interested in studying
best_tuple = [1000, 1000, 1000, 1000, 1000] # dummy data that is overwriten as the iterations run

for ind, degree in enumerate(degrees):
	print("> Running Ridge Regression for degree ", degree)

	# augmenting the input data to a given degree for both the train and validation sets
	polynome_train = build_poly(x_train,degree)
	polynome_validation = build_poly(x_validation, degree)

	lambdas = np.logspace(-15,0, num=50) # the range of lambdas we are interested in studying

	for lambda_ in lambdas:
		# For each lambda / degree combination, compute the MSEs and accuracy
		w, MSE_train = ridge_regression(y_train, polynome_train, lambda_)
		MSE_validation = compute_loss(y_validation, polynome_validation, w)
		y_pred = predict_labels(w, polynome_validation)
		y_acc = np.mean(y_validation == y_pred)

		# Keep track of the iteration with the lowest validation MSE
		if MSE_validation < best_tuple[1]:
			best_tuple = [MSE_train, MSE_validation, y_acc, degree, lambda_]

print("\nThe optimal iteration was computed to be the following:\nMSE_train: ", best_tuple[0], "\nMSE_validation: ", best_tuple[1], "\nAccuracy: ", best_tuple[2], "\nDegree:", best_tuple[3],"\nLambda:", best_tuple[4])

# Load the test data 
print("\nLoading test.csv...")
DATA_TEST_PATH = 'test.csv' 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# Augment the test input to the optimal degree
tX_test = build_poly(tX_test,best_tuple[3])
polynome_train = build_poly(x_train, best_tuple[3])

# replicate ridge regression using the optimal lambda value
w, MSE_train = ridge_regression(y_train, polynome_train, best_tuple[4])

print("Writing out.csv...\n")
OUTPUT_PATH = 'out.csv'
y_pred = predict_labels(w, tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
print(OUTPUT_PATH, " saved. You can upload this file at: https://www.aicrowd.com/challenges/epfl-machine-learning-higgs-2019 ")
