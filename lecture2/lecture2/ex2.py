#!/usr/bin/env python
# coding=utf-8
import math
import random

def load(filename):
    ret_X = []
    ret_y = []
    # ================== YOUR CODE HERE =============================

    # ===============================================================
    return ret_X, ret_y

def computeCost(X, y, theta0, theta1):
    J = 0
    # ================== YOUR CODE HERE =============================

    # ===============================================================
    return J

def gradientDescent(X, y, theta0, theta1, alpha, num_iters):
    m = len(X)
    for i in range(num_iters):
        pass
        # ================== YOUR CODE HERE =========================

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
