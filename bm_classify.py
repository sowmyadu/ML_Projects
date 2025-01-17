import numpy as np
import math

def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    #change y to -1
    """for i in range(N):
        if y[i] == 0:
            y[i] = -1"""
    
    y = np.where(y==0,-1,1)
    #print(y)
    
    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0 
        
        w_b = np.insert(w,0,0)
        X_b = np.insert(X,0,1,1)
        
        for i in range(max_iterations):
            prod = np.dot(w_b,np.transpose(X_b))
            #print(prod)
            #print("Perceptron: ",y*prod.shape)
            pred_fail = (y*prod<=0)
            fail = np.where(pred_fail)
            w_b = w_b + step_size * np.dot(y[fail], X_b[fail])/N
        b = w_b[0]
        w = w_b[-D:]
            
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0 
        
        w_b = np.insert(w,0,0)
        X_b = np.insert(X,0,1,1)
        
        for i in range(max_iterations):
            prod = sigmoid(y * np.dot(X_b,w_b))
            #p = sigmoid(np.matmul(X_b,w_b))
                            
            #gradient = (y-p)*p*(1-p)
            #gradient = np.dot(X_b.T,(prod - y))
            gradient = np.dot( (1-prod) * y,X_b)
            w_b = w_b + step_size * gradient/N
            #b = b + step_size * gradient * 1/N
        b = w_b[0]
        w = w_b[-D:]
        
        ############################################
        

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b


def sig(z):
    return 1.0/(1+np.exp(-z))

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = z
    z = np.array(z,dtype = np.float64)
    return 1.0/(1+np.exp(-z))
    #sigmoid_vec = np.vectorize(sig)
    #value = sigmoid_vec(z)
    
    ############################################
    
    #return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        for i in range(N):
            y_pred = np.dot(X[i],np.transpose(w)) + b
            if y_pred <= 0:
                y_pred = 0
            else:
                y_pred = 1
            preds[i] = y_pred 
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        for i in range(N):
            y_pred = sigmoid(np.dot(X[i],w.T) + b)
            if y_pred < 0.5:
                y_pred = 0
            else:
                y_pred = 1
            preds[i] = y_pred 
        ############################################
        

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds

def softmax_func(z):
    e_x = np.exp(z - np.max(z))
    e_sum = np.sum(e_x,axis =1)
    return e_x.T/e_sum

def softmax_func_gd(z):
    e_x = np.exp(z - np.max(z))
    e_sum = np.sum(e_x)
    return e_x/e_sum

def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        
        w_b = np.insert(w,0,0,1)
        #print(np.sum(X))
        X_b = np.insert(X,0,1,1)
        #print(w_b)
        #print(X_b)
        """for i in range(max_iterations):
            rand = np.random.choice(N)
            x_n = X_b[rand]
            y_n = y[rand]
            p = softmax(np.dot(x_n,w_b[np.dot(X_b,w_b))
            gradient = np.outer(p,X[n])
            w_b = w_b - step_size * np.dot(p-1,X_b)"""
        
        """for i in range(max_iterations):
            rand = np.random.choice(N)
            xn = X_b[rand]
            yn = y[rand]
            p = softmax_func(np.matmul(np.matrix(xn),np.matrix(w_b).T))
            w_b = w_b - step_size * np.outer((p.T), xn)"""
        
        for i in range(max_iterations):
            rand = np.random.choice(N)
            xn = X[rand]
            yn = y[rand]
            p = softmax_func_gd(np.add(np.dot(w,xn),b))
            #p = np.matrix(p)
            #print(p)
            p[yn] = p[yn] - 1
            #print(np.matrix(xn).shape)
            #w_b = w_b - step_size * np.matmul(np.matrix(p).T,np.matrix(xn))
            #print(p.shape,":::",xn.shape,":::",w.shape)
            w = w - step_size * np.outer(p,xn)
            b = b - step_size * p
        #b = w_b[:,0]
        #w = w_b[:,1:D+1]
        #print(b.shape, "::", N, "::", C,"::", D)
        ############################################
        

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        
        w_b = np.insert(w,0,0,1)
        X_b = np.insert(X,0,1,1)
        
        
        #change y
        y_enc = np.zeros((N,C))
        y_enc[np.arange(N),y] = 1
        #print(y_enc)
        
        #max_iterations = 2
        for i in range(max_iterations):
            p = softmax_func(np.dot(X_b, w_b.T))
            mu = y_enc-p.T
            
            gradient = np.dot(mu.T,X_b)
            w_b = w_b + step_size * gradient/N
        b = w_b[:,0]
        w = w_b[:,1:D+1]
        ############################################
        

    else:
        raise "Type of Gradient Descent is undefined."
    
    #print(w.shape,":: ", C, ":: ", D)
    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b

def softmax(X, w, b):
    z = np.add(np.dot(X,w.T),b)
    exp = np.exp(z)
    y = exp.T/np.sum(exp,axis = 1)
    return y.T
    
def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    preds = np.zeros(N)
    y = softmax(X,w, b)
    y_pred = np.argmax(y, axis = 1)
    preds = y_pred
    
    ############################################

    assert preds.shape == (N,)
    return preds




        