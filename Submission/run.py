# Authors: 
# Daniel-Florin Dosaru
# Natasha Ørregaard Klingenbrunn
# Sena Necla Cetin

import numpy as np
import matplotlib.pyplot as plt
import csv

# This function randomly splits the training data into train - test sets using a certain ratio and 
def split_data(x, y, ratio, seed=1):
	np.random.seed(seed)

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

#This function is used for feature augmentation using a certain
def build_poly(x, degree):
    polynome = x
    for d in range(2, degree+1):
        polynome = np.concatenate((polynome, np.power(x[:,1::], d)), axis = 1)
    return polynome


# Function from proj1_helper.py
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

# Function from proj1_helper.py
def predict_labels(weights, data):
	"""Generates class predictions given weights, and a test data matrix"""

	y_pred = np.dot(data, weights)
	y_pred[np.where(y_pred <= 0)] = -1
	y_pred[np.where(y_pred > 0)] = 1

	return y_pred

# Function from proj1_helper.py
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



# Mean Squared Error
def compute_loss(y, tx, w):
	MSE = 1/(2*y.shape[0])*np.sum(np.square(y-np.dot(tx,w)))
	return MSE

# Ridge regression implementation
def ridge_regression(y, tx, lambda_):
	a = (1/len(y))*(tx.T @ tx) + 2*(lambda_*np.identity(tx.shape[1]))
	b = (1/len(y))*(tx.T @ y)
	w = np.linalg.solve(a,b)
	MSE = compute_loss(y, tx, w)

	return w, MSE


# Load the data
print("Loading training data...")
DATA_TRAIN_PATH = 'train.csv' # TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

print("Splitting training data...")
# Split training data 
ratio = 0.8
seed = 1
x_train, x_validation, y_train, y_validation = split_data(tX, y, ratio, seed)

print("Running ridge regression...")
# Running ridge regression
degrees = [1, 2, 3, 4, 5, 6, 7]
x_train, x_validation, y_train, y_validation = split_data(tX, y, ratio, seed)

# dummy data that is overwriten as the iterations run
best_tuple = [1000, 1000, 1000, 1000, 1000]

for ind, degree in enumerate(degrees):
	print("\tRunning ridge regression polynomial degree = ", degree)

	# augmenting the input data to a given degree
	polynome_train = build_poly(x_train,degree)
	polynome_validation = build_poly(x_validation, degree)

	lambdas = np.logspace(-15,0, num=10)
	plot_lost = []

	for lambda_ in lambdas:
		w, MSE_train = ridge_regression(y_train, polynome_train, lambda_)
		MSE_validation = compute_loss(y_validation, polynome_validation, w)
		y_pred = predict_labels(w, polynome_validation)
		y_acc = np.mean(y_validation == y_pred)

		print([MSE_train, MSE_validation, y_acc, degree, lambda_])
		if MSE_validation < best_tuple[1]:
			best_tuple = [MSE_train, MSE_validation, y_acc, degree, lambda_]

		plot_lost.append((MSE_train, MSE_validation, y_acc, degree, lambda_))

l1, l2, l3, l4, l5 = zip(*plot_lost)

print("Computed best tuple of MSE_train, MSE_validation, y_acc, degree, lambda_:\n", best_tuple)


print("Loading test.csv...")
DATA_TEST_PATH = 'test.csv' # TODO: unzip the file
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

tX_test = build_poly(tX_test,6)
polynome_train = build_poly(x_train,6)
w, MSE_train = ridge_regression(y_train, polynome_train, 1e-15)


print("Writing out.csv...")
OUTPUT_PATH = 'out.csv'
y_pred = predict_labels(w, tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
print(OUTPUT_PATH, " saved. You can upload this file at: https://www.aicrowd.com/challenges/epfl-machine-learning-higgs-2019 ")
