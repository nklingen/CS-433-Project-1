
import numpy as np
import csv

# This function randomly splits the training data into train - test sets using a certain ratio and seed
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



# Generate a minibatch iterator for a dataset.
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
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

#This function is used for feature augmentation using a certain degree
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

def predict_labels_log(weights, data):

	y_pred = np.dot(data, weights)
	y_pred[np.where(y_pred <= 0.5)] = -1
	y_pred[np.where(y_pred > 0.5)] = 1

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
