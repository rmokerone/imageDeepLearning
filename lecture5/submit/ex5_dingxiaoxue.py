#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from scipy import ndimage

def load_dataset():
    train_set_x_orig = np.zeros((1,1))
    train_set_y = np.zeros((1, 1))
    test_set_x_orig = np.zeros((1, 1))
    test_set_y = np.zeros((1, 1))
    classes = np.zeros((1, 1))
    # ===================== YOUR CODE HERE =====================================
    datasets1=h5py.File('train_catvnoncat.h5','r')
    datasets2=h5py.File('test_catvnoncat.h5','r')
    train_set_x_orig=datasets1["train_set_x"][:]
    train_set_y=datasets1["train_set_y"][:].reshape(len(datasets1["train_set_y"][:]),1).T
    test_set_x_orig =datasets2["test_set_x"][:]
    test_set_y = datasets2["test_set_y"][:].reshape(len(datasets2["test_set_y"][:]),1).T
    classes = datasets2['list_classes'][:]
    # ==========================================================================
    return train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes

def imshow(train_set_x_orig, idx):
    # ===================== YOUR CODE HERE =====================================
    plt.imshow(train_set_x_orig[idx])
    plt.show()
    # ==========================================================================

def flatten_shape(train_set_x_orig, test_set_x_orig):
    train_set_x_flatten = train_set_x_orig
    test_set_x_flatten = test_set_x_orig
    # ===================== YOUR CODE HERE =====================================
    train_set_x_flatten=train_set_x_orig.reshape(len(train_set_x_orig),64*64*3).T
    test_set_x_flatten =test_set_x_orig.reshape(len(test_set_x_orig),64*64*3).T
    # ==========================================================================
    return train_set_x_flatten, test_set_x_flatten

def feature_scaling(vals):
    norm_vals = vals
    # ===================== YOUR CODE HERE =====================================
    norm_vals=vals/255
    # ==========================================================================
    return norm_vals

def sigmoid(z):
    s = z
    # ===================== YOUR CODE HERE =====================================
    s = 1 / (1 + np.exp(-z))
    # ==========================================================================
    return s

def initialize_with_zeros(dim):
    w = np.zeros((1, 1))
    b = np.zeros((1, 1))
    # ===================== YOUR CODE HERE =====================================
    w=np.zeros((dim,1))
    b=0.0
    # ==========================================================================
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]
    
    dw = w
    db = np.array([[0.0]])
    cost = np.zeros((1, 1))
    # ===================== YOUR CODE HERE =====================================
    A = sigmoid(np.dot(w.T,X)+b)
    cost=-(np.dot(Y,np.log(A).T)+np.dot((1-Y),np.log(1-A).T))/m
    dw = np.dot(X,(A-Y).T)/m
    db = np.sum(A-Y)/m
    # ==========================================================================
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {"dw": dw,
             "db": db}

    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    dw = w
    db = b
    for i in range(num_iterations):
    # ===================== YOUR CODE HERE =====================================
        k,cost=propagate(w, b, X, Y)
        costs.append(cost)
        dw=k["dw"]
        db=k["db"]
        w = w-learning_rate*dw
        b = b-learning_rate*db
    # ==========================================================================
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    # ===================== YOUR CODE HERE =====================================
    A = sigmoid(np.dot(w.T,X)+b)
    for i in range(m):
        if(A[0][i]>0.5):
            Y_prediction[0][i]=1
        else:
            Y_prediction[0][i]=0
    # ==========================================================================
    assert(Y_prediction.shape == (1, m))
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    return d

def plot(d):
    costs = d["costs"]
    # ===================== YOUR CODE HERE =====================================
    plt.xlabel('iterations(per hundreds)')  
    plt.ylabel('cost')  
    plt.title('learning rate=0.005') 
    plt.plot(costs)   
    plt.show() 
    # ==========================================================================

if __name__ == "__main__":
    idx = 2
    # ======================== Part 1: Loading Data ============================
    print('Loading Data ...')
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    print("The shape of the variable train_set_x_orig is", train_set_x_orig.shape)
    print("The shape of the variable train_set_y is", train_set_y.shape)
    print("The shape of the variable test_set_x_orig is", test_set_x_orig.shape)
    print("The shape of the variable test_set_y is", test_set_y.shape)
    print("Expected shape of the variables are:")
    print("(209, 64, 64, 3)")
    print("(1, 209)")
    print("(50, 64, 64, 3)")
    print("(1, 50)")
    print("(1, 2)")
    # ======================== Part 2: Displaying Image ========================
    print('Displaying Image ...')
    imshow(train_set_x_orig, idx)
    # ======================== Part 3: Flatten Shape ===========================
    print('Flatten Shape ...')
    train_set_x_flatten, test_set_x_flatten = flatten_shape(train_set_x_orig, test_set_x_orig)
    print("The shape of the variable train_set_x_flatten is", train_set_x_flatten.shape)
    print("The shape of the variable test_set_x_flatten is", test_set_x_flatten.shape)
    print("Expected shape of the variables are:")
    print("(12288, 209)")
    print("(12288, 50)")
    # ======================== Part 4: Feature Scaling =========================
    print("Feature Scaling ...")
    train_set_x = feature_scaling(train_set_x_flatten)
    test_set_x = feature_scaling(test_set_x_flatten)
    print("train_set_x[:3, 0] = ", train_set_x[:3, 0])
    print("Expected train_set_x[:3, 0] =  [ 0.06666667  0.12156863  0.21960784]")
    print("test_set_x[:3, 0] = ", test_set_x[:3, 0])
    print("Expected test_set_x[:3, 0] =  [ 0.61960784  0.40784314  0.3254902 ]")
    # ======================== Part 5: Sigmoid Function ========================
    print("Checking Sigmoid ...")
    test_sigmoid = sigmoid(np.array([[0, 2]]))
    print("test_sigmoid = ", test_sigmoid)
    print("Expected test_sigmoid =  [[ 0.5         0.88079708]]")
    # ======================== Part 6: Initialize Parameters ===================
    print("Initialize Parameters ....")
    dim = train_set_x.shape[0]
    w, b = initialize_with_zeros(dim)
    print("The shape of the variable w is", w.shape)
    print("Expected shape of the variable w is (%d, 1)" % dim)
    print("b = ", b)
    print("Expected b = 0.0")
    #J = compute_cost(train_set_x, train_set_y, )
    # ======================== Part 7: Propagate ===============================
    print("Checking Propagate ...")
    w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
    grads, cost = propagate(w, b, X, Y)
    print("dw = " + str(grads["dw"]))
    print('Expected dw = [[ 0.99845601]     [ 2.39507239]])')
    print("db = " + str(grads["db"]))
    print('Expected db = 0.00145557813678')
    print("cost = " + str(cost))
    print('Expected cost = 5.801545319394553')
    # ======================== Part 8: Parameter optimization ==================
    print("Parameter optimization ...")
    params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)
    print("w = " + str(params["w"]))
    print("Expected w = [[ 0.19033591]      [ 0.12259159]]")
    print("b = " + str(params["b"]))
    print("Expected b = 1.92535983008")
    print("dw = " + str(grads["dw"]))
    print("Expected dw = [[ 0.67752042]     [ 1.41625495]]")
    print("db = " + str(grads["db"]))
    print("Expected db = 0.219194504541")
    # ======================== Part 9: Predict =================================
    print("Checking Predict ...")
    w = np.array([[0.1124579],[0.23106775]])
    b = -0.3
    X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
    print("predictions = " + str(predict(w, b, X)))
    print("Expected predictions = [[1 1 0]]")
    # ======================== Part 10: Runing model ============================
    print("Runing Model ...")
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
    # ======================== Part 11: Plot Learning curve ====================
    print("Plot Learning curve ...")
    plot(d)
    # ======================== Part 12: Test your image ========================
    my_image = "squirrel.jpg"   # change this to the name of your image file
    num_px = 64
    fname = "images/" + my_image
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
    my_predicted_image = predict(d["w"], d["b"], my_image)
 
    plt.imshow(image)
    
    #print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    print("y = " + str(np.squeeze(my_predicted_image)))
    if np.squeeze(my_predicted_image):
        print("your algorithm predicts a cat")
    else:
        print("Your algorithm predicts a noncat")
    plt.show()