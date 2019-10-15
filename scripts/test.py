from proj1_helpers import *

DATA_TRAIN_PATH = '../data/train.csv' # TODO: download train data and supply path here
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

def compute_loss(y, tx, w):
    """Calculate the loss using MSE"""
    # ***************************************************
    # MSE = 1/(2*y.shape[0])*np.sum(np.square(y-np.dot(tx,w)))

    N = np.shape(y)[0]
    e = y - np.dot(tx, w)
    loss = (1/(2*N))*np.dot(e, np.transpose(e))
    return MSE

def main():
    print(tX.shape)
    print(compute_loss(y,tX, np.ones((30, 1))))
    print("H1123")

if __name__== "__main__":
    main()
