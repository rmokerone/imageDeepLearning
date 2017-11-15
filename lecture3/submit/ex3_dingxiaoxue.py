#!/usr/bin/env python
# coding=utf-8
import math
import random

def load(filename):
    ret_X1 = []
    ret_X2 = []
    ret_y = []
    # ================== YOUR CODE HERE =============================
    for line in open(filename):
        l=line.split(',')
        ret_X1.append(float(l[0]))
        ret_X2.append(float(l[1]))
        ret_y.append(float(l[2]))
    # ===============================================================
    return ret_X1,ret_X2, ret_y

def computeCost(X1,X2, y, theta0, theta1,theta2):
    J = 0
    # ================== YOUR CODE HERE =============================
    sum=0
    for i in range(m):
        sum+=(theta0+theta1*X1[i]+theta2*X2[2]-y[i])**2
    J=sum/(2*m)
    # ===============================================================
    return J

def gradientDescent(X1,X2, y, theta0, theta1, theta2, alpha, num_iters):
    m = len(X1)
    for i in range(num_iters):
        # ================== YOUR CODE HERE =========================
        sum1=0
        sum2=0
        sum3=0
        for i in range(m):
            sum1+=(theta0+theta1*X1[i]+theta2*X2[i]-y[i])
            sum2+=(theta0+theta1*X1[i]+theta2*X2[i]-y[i])*X1[i]
            sum3+=(theta0+theta1*X1[i]+theta2*X2[i]-y[i])*X2[i]
        theta0=theta0-alpha*sum1/m
        theta1=theta1-alpha*sum2/m
        theta2=theta2-alpha*sum3/m
        # ===========================================================
    return theta0, theta1,theta2


if __name__ == '__main__':
    # initialization parameter
    theta0 = 0.0
    theta1 = 0.0
    theta2 = 0.0
    iterations = 3000
    alpha = 0.01
    # ================ Part 1: Loading Data =========================
    print('Loading Data ...')
    X1,X2, y = load('c:\\dataset_2_2.txt')
    m = len(y)
    print('m = %s' % m)
    print('X1[:10] = %s' % X1[:10])
    print('X2[:10] = %s' %X2[:10])
    print('y[:10] = %s' % y[:10])

    # ================ Part 2: Testing CostFuntion ==================
    print('Testing CostFuntion ...')
    # compute initial cost
    J = computeCost(X1,X2, y, theta0, theta1,theta2)
    print('With theta0 = 0 and theta1 = 0 and theta2 = 0')
    print('Cost computed = %f' % J)
    #print('Expected cost value (approx) 32.07')
    # further testing of the cost function
    J = computeCost(X1,X2, y, 0.0, 1.0,1.0)
    print('with theta0 = 0 and theta1 = 1 and theta2 = 1')
    print('Cost computed = %f' % J)
    #print('Expected cost value (approx) 54.24')

    # ================ Part 3: Gradient descent =====================
    print('Running Gradient Descent ...')
    theta0, theta1,theta2 = gradientDescent(X1,X2, y, theta0, theta1,theta2, alpha, iterations)
    print('Theta found by gradient descent:')
    print('theta0 = %f, theta1 = %f,theta3 = %f' % (theta0, theta1,theta2))
    print('Expected theta values (approx)')
    #print('theta0 = -3.6303, theta1 = 1.1664')

    # ================ Part 4: Predict values =======================
    #predict1 = theta0 + 3.5 * theta1
    #print('For population = 35000, we predict a profit of %.2f' % (predict1 * 10000))
    #predict2 = theta0 + 7.0 * theta1
    #print('For population = 70000, we predict a profit of %.2f' % (predict2 * 10000))
