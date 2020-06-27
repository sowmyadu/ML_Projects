"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #print("Weight: ",w)
    y_pred = np.dot(X, w)
    y_predT = y_pred.transpose()
    n = np.size(X,0)
    mae_sum = 0
    for i in range(n):
        mae_sum += np.absolute(y_pred[i] - y[i])
    mae = mae_sum/n
    
    #print(X)
    #print(y)
    
    
    #####################################################
    err = mae
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
    """
      Compute the weight parameter given X and y.
      Inputs:
      - X: A numpy array of shape (num_samples, D) containing feature.
      - y: A numpy array of shape (num_samples, ) containing label
      Returns:
      - w: a numpy array of shape (D, )
    """
    
    #####################################################
    #	TODO 2: Fill in your code here #
    #w = (xTx)-1XTy
    #dot_xy = np.dot(X_T, y)  
    #####################################################
    #print("No. of cols of X: ", np.size(X, 1))
    #print("No. of rows: ", np.size(X, 0))
    X_T = X.transpose()
    array = np.dot(X_T, X)
    #print("No. of cols: ",np.size(array, 1))
    #print("No. of rows: ", np.size(array, 0))
    array_inv = np.linalg.inv(array)
    y_array = np.dot(X_T, y)
    w = np.dot(array_inv, y_array)
    return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    
    X_T = X.transpose()
    array = np.dot(X_T, X)
    eigenvals = np.linalg.eigvals(array)
    #print(eigenvals)
    array_min = np.amin(np.absolute(eigenvals))
    #print(array_min)
    final_arr = array
    while array_min ==0:
        u,v = array.shape
        id_mat = np.dot(np.identity(u),0.1)
        final_arr = np.add(array, id_mat)
        #print(final_arr)
        eigenvals = np.linalg.eigvals(final_arr)
        array_min = np.amin(np.absolute(eigenvals))
        #print(eigenvals)
        #print(array_min)
    array_inv = np.linalg.inv(final_arr)
    y_array = np.dot(X_T, y)
    w = np.dot(array_inv, y_array)
    return w
    
    
    #####################################################


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
    
    X_T = X.transpose()
    array = np.dot(X_T, X)
    u,v = np.shape(array)
    id_mat = np.dot(np.identity(u),lambd)
    final_arr = np.add(array,id_mat)
    array_inv = np.linalg.inv(final_arr)
    y_array = np.dot(X_T, y)
    w = np.dot(array_inv, y_array)
    
  #####################################################		
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    
    lambd = 1.0e-19
    min_err = 1
    ctr = 20
    while lambd <= 1.0e+19:
        ctr = ctr - 1
        w = regularized_linear_regression(Xtrain,ytrain,lambd)
        err = mean_absolute_error(w,Xval,yval)
        if err < min_err:
            bestlambda = lambd
            min_err = err
            best_ctr = ctr
        lambd = lambd*10.0
    #####################################################		
    #bestlambda = None
    bestlambda = round(bestlambda, best_ctr)
    return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    X_fin = X
    for i in range(2,power+1):
        X_pow = np.power(X,i)
        X_fin = np.append(X_fin,X_pow, axis=1)
        
    #####################################################		
    
    return X_fin


