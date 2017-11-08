#!/usr/bin/env python
# coding=utf-8
import math
import random

def load(filename):
    ret_X = []
    ret_y = []
    with open(filename,'r') as f:
        for line in f.readlines():
            p = line.strip().split(',')
            L = list(map(float,list(p)))
            ret_X.append(L[0])
            ret_y.append(L[1])
        return ret_X, ret_y

def computeCost(X, y, theta0, theta1):
    J = 0
    n = len(X)
    for i in range(n):              #求和 H(Xi)-Yi
        b = theta0 + X[i]*theta1 - y[i]
        J = J+b*b
    J = J/(2*m)                     #cost function 求和J/2m
    return J

def gradientDescent(X, y, theta0, theta1, alpha, num_iters):
    m = len(X)
    for i in range(num_iters):
        J0 = J1 = 0
        for j in range(m):              #单独分别求temp0、temp1的和公式对应（temp1乘Xi）
            b = theta0 + theta1*X[j] - y[j]
            J0 = J0 + b
            J1 = J1 + b*X[j]
        temp0 = theta0 - alpha*J0/m     #带入指定计算下降公式，对应正确迭代步骤
        temp1 = theta1 - alpha*J1/m
        theta0 = temp0
        theta1 = temp1
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
