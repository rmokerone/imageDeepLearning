import math
import random

def load(filename):
    ret_x1 = []
    ret_x2 = []
    ret_y = []
    # ================== YOUR CODE HERE =============================
    A = open('F:/Work/lecture1/lecture1/dataset_1_1.txt','r')
    while True:
        x = A.readline()
        if x=='':
            break
        l=x.split()
        ret_x1.append(float(l[0])/100) 
        ret_x2.append(float(l[1])/100) 
    B = open('F:/Work/lecture1/lecture1/output_1_1.txt','r')
    while True:
        l = B.readline()
        if l == '':
            break
        ret_y.append(float(l)/100) 
    # ===============================================================
    return ret_x1, ret_x2, ret_y

def computeCost(x1,x2, y, theta0, theta1,theta2):
    J = 0
    # ================== YOUR CODE HERE =============================
    m = len(x1)
    sum = 0
    d = list(range(m))
    for i in d:
        h = theta0 + theta1 * x1[i] + theta2 * x2[i]
        sum += (h - y[i]) * (h - y[i])
    J = sum / 2 /m
    # ===============================================================
    return J

def gradientDescent(x1,x2,y,theta0, theta1,theta2,alpha,num_iters):
    m = len(y)
    for i in range(num_iters):
        pass
        # ================== YOUR CODE HERE =========================
        sum0 = 0
        sum1 = 0
        sum2 = 0
        d = list(range(m))
        for j in d:
            h = theta0 + theta1 * x1[j] + theta2 * x2[j]
            sum0 = sum0 + (h - y[j])
            sum1 = sum1 + (h - y[j]) * x1[j]
            sum2 = sum2 + (h - y[j]) * x2[j]
        theta0 = theta0 - alpha / m * sum0
        theta1 = theta1 - alpha / m * sum1
        theta2 = theta2 - alpha / m * sum2
        # ===========================================================
    return theta0, theta1,theta2

if __name__ == '__main__':
    # initialization parameter
    theta0 = 0.0
    theta1 = 0.0
    theta2 = 0.0
    iterations = 3000
    alpha = 0.032
    # ================ Part 1: Loading Data =========================
    x1,x2, y = load('dataset_3_1.txt')
    m = len(y)   
    # ================ Part 2: Testing CostFuntion ==================
    # compute initial cost
    J = computeCost(x1,x2, y, theta0, theta1,theta2)
    # further testing of the cost function
    J = computeCost(x1,x2, y, 0,1, 1)

    # ================ Part 3: Gradient descent =====================
    print('Running Gradient Descent ...')
    theta0,theta1,theta2 = gradientDescent(x1,x2,y,theta0,theta1,theta2,alpha, iterations)
    print('Theta found by gradient descent:')
    print('theta0 = %.6f, theta1 = %.6f,theta2 = %.6f ' % (theta0, theta1,theta2))

    