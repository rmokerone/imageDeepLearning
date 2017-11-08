#!/usr/bin/env python
# coding=utf-8
import math
import random

def load(filename):
    ret_X = []
    ret_y = []
    # ================== YOUR CODE HERE =============================
    A=open('F:/Work/lecture2/lecture2/dataset_2_1.txt','r')
    while True:
        x=A.readline()
        if x=='':
            break
        l=x.split(',')
        ret_X.append(float(l[0]))
        ret_y.append(float(l[1]))
    # ===============================================================
    return ret_X, ret_y

def computeCost(X, y, theta0, theta1):
    J = 0
    # ================== YOUR CODE HERE =============================
    m = len(X)
    sum = 0
    d = list(range(m))
    for i in d:
        h = theta0 + theta1 * X[i]
        sum = sum + (h - y[i]) * (h - y[i])
    J = sum / 2 /m
    # ===============================================================
    return J

def gradientDescent(X, y, theta0, theta1, alpha, num_iters):
    m = len(X)
    for i in range(num_iters):
        pass
        # ================== YOUR CODE HERE =========================
        sum0 = 0
        sum1 = 0
        d = list(range(m))
        for j in d:
            h = theta0 + theta1 * X[j]
            sum0 = sum0 + (h - y[j])
            sum1 = sum1 + (h - y[j]) * X[j]
        theta0 = theta0 - alpha / m * sum0
        theta1 = theta1 - alpha / m * sum1
        # ===========================================================
    return theta0, theta1


if __name__ == '__main__':
    # initialization parameter
    theta0 = 0.0
    theta1 = 0.0
    iterations = 1500
    alpha = 0.01
    # ================ Part 1: Loading Data =========================
    print('Loading Data ...')
    X, y = load('dataset_2_1.txt')
    m = len(y)
    print('m = %s' % m)
    print('X[:10] = %s' % X[:10])
    print('y[:10] = %s' % y[:10])

    # ================ Part 2: Testing CostFuntion ==================
    print('Testing CostFuntion ...')
    # compute initial cost
    J = computeCost(X, y, theta0, theta1)
    print('With theta0 = 0 and theta1 = 0')
    print('Cost computed = %f' % J)
    print('Expected cost value (approx) 32.07')
    # further testing of the cost function
    J = computeCost(X, y, -1.0, 2.0)
    print('with theta0 = -1 and theta1 = 2')
    print('Cost computed = %f' % J)
    print('Expected cost value (approx) 54.24')

    # ================ Part 3: Gradient descent =====================
    print('Running Gradient Descent ...')
    theta0, theta1 = gradientDescent(X, y, theta0, theta1, alpha, iterations)
    print('Theta found by gradient descent:')
    print('theta0 = %f, theta1 = %f' % (theta0, theta1))
    print('Expected theta values (approx)')
    print('theta0 = -3.6303, theta1 = 1.1664')

    # ================ Part 4: Predict values =======================
    predict1 = theta0 + 3.5 * theta1
    print('For population = 35000, we predict a profit of %.2f' % (predict1 * 10000))
    predict2 = theta0 + 7.0 * theta1
    print('For population = 70000, we predict a profit of %.2f' % (predict2 * 10000))
