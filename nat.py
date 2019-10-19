#%% [markdown]
# <a href="https://colab.research.google.com/github/nklingen/CS-433-Project-1/blob/master/implementations.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

#%%
import numpy as np


#%%
from google.colab import files
uploaded = files.upload()


#%%
def least_squares_GD(y, tx, initial_w, max_iters, gamma): # Linear regression using gradient descent


#%%
def least_squares_SGD(y, tx, initial_w, max_iters, gamma): # Linear regression using stochastic gradient descent
  return (w, loss) # return last weight vector of the method and the corresponding loss value (cost function)


#%%
def least_squares(y, tx):
    w = np.linalg.solve(np.dot(tx.T,tx), np.dot(tx.T,y))
    MSE = 1/(2*y.shape[0])*np.sum(np.square(y-np.dot(tx,w)))
    return MSE, w

#%%
def ridge_regression(y, tx, lambda_):
    a = (1/len(y))*(np.dot(tx.T,tx)) + 2*(lambda_*np.identity(tx.shape[1]))
    b = (1/len(y))*np.dot(tx.T,y)
    weight = np.linalg.solve(a,b)
    return weight

#%%
def logistic_regression(y, tx, initial_w, max_iters, gamma):


#%%
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):


#%%
# calculate the correlation matrix and eliminate the features that are very correlated with each other or create new features

