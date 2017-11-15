#!/usr/bin/env python
# coding=utf-8
import math
import random

def load(filename):
    ret_X1 = []
    ret_X2 = []
    ret_y = []
    a1=[]
    a2=[]
    # ================== YOUR CODE HERE =============================
    for line in open(filename):
        l=line.split(' ')
        a1.append(int(l[0]))
        a2.append(int(l[1]))
    s1=max(a1)-min(a1)
    s2=max(a2)-min(a2)
    a1aver=float(sum(a1)/len(a1))
    a2aver=float(sum(a2)/len(a2))
    m=len(a1)
    for k in range(m):
        ret_X1.append(float((a1[k]-a1aver)/s1))
        ret_X2.append(float((a2[k]-a2aver)/s2))
        ret_y.append(float((a1[k]-a1aver)/s1+(a2[k]-a2aver)/s2))
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
    alpha = 0.03
    # ================ Part 1: Loading Data =========================
    print('Loading Data ...')
    X1,X2, y = load('c:\\dataset_1_1.txt')
    m = len(y)
    print('m = %s' % m)
    print('X1[:10] = %s' % X1[:10])
    print('X2[:10] = %s' %X2[:10])
    print('y[:10] = %s' % y[:10])

    # ================ Part 2: Testing CostFuntion ==================
    print('Testing CostFuntion ...')
    J = computeCost(X1,X2, y, theta0, theta1,theta2)
    print('With theta0 = 0 and theta1 = 0 and theta2 = 0')
    print('Cost computed = %f' % J)
    J = computeCost(X1,X2, y, 0.0, 1.0,1.0)
    print('with theta0 = 0 and theta1 = 1 and theta2 = 1')
    print('Cost computed = %f' % J)


    # ================ Part 3: Gradient descent =====================
    print('Running Gradient Descent ...')
    theta0, theta1,theta2 = gradientDescent(X1,X2, y, theta0, theta1,theta2, alpha, iterations)
    print('Theta found by gradient descent:')
    print('theta0 = %f, theta1 = %f,theta3 = %f' % (theta0, theta1,theta2))
   

    # ================ Part 4: Predict values =======================
    #predict1 = theta0 + 3.5 * theta1
    #print('For population = 35000, we predict a profit of %.2f' % (predict1 * 10000))
    #predict2 = theta0 + 7.0 * theta1
    #print('For population = 70000, we predict a profit of %.2f' % (predict2 * 10000))
