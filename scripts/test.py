from proj1_helpers import *

DATA_TRAIN_PATH = '../data/train.csv' # TODO: download train data and supply path here
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

def compute_loss_v1(y, tx, w):
    """Calculate the loss using MSE v1: natasha"""
    MSE = 1/(2*y.shape[0])*np.sum(np.square(y-np.dot(tx,w)))
    return MSE

def compute_loss_v2(y, tx, w):
    """Calculate the loss using MSE v2 :daniel"""

    N = np.shape(y)[0]
    e = y - np.dot(tx, w)
    return (1/(2*N))*np.dot(e, np.transpose(e))

def main():
    w = np.ones(30)
    print(compute_loss_v1(y, tX, w))

    print(compute_loss_v2(y, tX, w))


if __name__== "__main__":
    main()
