#!/usr/bin/env python
# coding=utf-8
import math
import random

def load(filename):
    ret_X1 = []
    ret_X2 = []
    ret_y = []
    with open(filename,'r') as f:
        for line in f.readlines():
            p = line.strip().split(' ')
            L = list(map(float,p))
            ret_X1.append(L[0]/300)
            ret_X2.append(L[1]/300)
            ret_y.append(L[2]/300)
        return ret_X1, ret_X2, ret_y

def computeCost(X1, X2 , y, theta0, theta1, theta2):
    J = 0
    n = len(X1)
    for i in range(n):              #求和 H(Xi)-Yi
        b = theta0 + X1[i]*theta1 + X2[i]*theta2 - y[i]
        J = J+b*b
    J = J/(2*m)                     #cost function 求和J/2m
    return J

def gradientDescent(X1, X2, y, theta0, theta1, theta2, alpha, num_iters):
    m = len(X1)
    for i in range(num_iters):
        J0 = J1 = J2 = 0
        for j in range(m):              #单独分别求temp0、temp1、temp2的和
            b = theta0 + theta1*X1[j] +theta2*X2[j]- y[j]
            J0 = J0 + b
            J1 = J1 + b*X1[j]
            J2 = J2 + b*X2[j]
        temp0 = theta0 - alpha*J0/m     #带入指定计算下降公式，对应正确迭代步骤
        temp1 = theta1 - alpha*J1/m
        temp2 = theta2 - alpha*J2/m
        theta0 = temp0
        theta1 = temp1
        theta2 = temp2
    return theta0, theta1, theta2


if __name__ == '__main__':
    # initialization parameter
    theta0 = 0.0
    theta1 = 0.0
    theta2 = 0.0
    iterations = 1200
    alpha = 0.1
    # ================ Part 1: Loading Data =========================
    print('Loading Data ...')
    X1, X2, y = load('1.txt')
    m = len(y)
    print('m = %s' % m)
    print('X1[:10] = %s' % X1[:10])
    print('X2[:10] = %s' % X2[:10])
    print('y[:10] = %s' % y[:10])

    # ================ Part 2: Testing CostFuntion ==================
    print('Testing CostFuntion ...')
    # compute initial cost
    J = computeCost(X1, X2, y, theta0, theta1, theta2)
    print('With theta0 = 0 and theta1 = 0 and theta2 = 0')
    print('Cost computed = %f' % J)
    print('Expected cost value (approx) 6.80')
    # further testing of the cost function
    J = computeCost(X1, X2, y, -1.0, 2.0, 4.0)
    print('with theta0 = -1 and theta1 = 2 and theta2 = 4')
    print('Cost computed = %f' % J)
    print('Expected cost value (approx) 22.10')

    # ================ Part 3: Gradient descent =====================
    print('Running Gradient Descent ...')
    theta0, theta1, theta2  = gradientDescent(X1, X2, y, theta0, theta1, theta2, alpha, iterations)
    print('Theta found by gradient descent:')
    print('theta0 = %f, theta1 = %f, theta2 = %f' % (theta0, theta1,theta2))
    print('Expected theta values (approx)')
    print('theta0 = 0.000000, theta1 = 1.000000 ，theta2 = 1.000000')

    # ================ Part 4: Predict values =======================
    predict1 = theta0 + 3.5 * theta1 + 3.5*theta2
    print('For population = 35000, we predict a profit of %.2f' % (predict1*10000 ))
    predict2 = theta0 + 7.0 * theta1 + 7.0*theta2
    print('For population = 70000, we predict a profit of %.2f' % (predict2*10000 ))
