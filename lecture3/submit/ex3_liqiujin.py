#!/usr/bin/python
# -*- coding: utf-8 -*-
import math 
import random

def load():
    ret_X1 = []
    ret_X2 = []
    ret_Y = []
    m = 0
    f = open('C:\\Users\Lqj\Desktop\jiqixuexi(jiang)\lecture3\dataset_3_1.txt','r')
    Data1 = f.readlines()
    for ele1 in Data1 :
        element1 = ele1.strip().split()
        m = m+1
        ret_X1.append(float(element1[0]))
        ret_X2.append(float(element1[1]))
    print('m =',m)
    f =open('C:\\Users\Lqj\Desktop\jiqixuexi(jiang)\lecture3\output_3_1.txt','r')
    Data2 = f.readlines()
    for ele2 in Data2 :
        element2 = ele2.strip()
        ret_Y.append(float(element2))
    return ret_X1, ret_X2, ret_Y

def ComputeCost(x1, x2, y, theta0,theta1,theta2):
    i = 0
    sum = 0 
    while i < m :
        a = x1[i]
        b = x2[i]
        c = y[i]
        h = theta0 + theta1*a + theta2*b
        sum = sum + (h-c)*(h-c)
        i = i + 1
    J = sum / (2*m)
    return J
def gradientDescent(x1, x2, y, theta0, theta1, theta2,alpha, num_iters):
    for i in range(num_iters):
        pass
    sum1 = 0
    sum2 = 0
    sum3 = 0
    k = 0
    while k < m:
        a = x1[k]
        b = x2[k]
        c = y[k]
        n = theta0 + theta1*a + theta2*b
        sum1 = sum1 +(n-c)*1.0
        sum2 = sum2 +(n-c)*a
        sum3 = sum3 +(n-c)*b
        k = k + 1
        theta0 = theta0-alpha*(sum1/m)
        theta1 = theta1-alpha*(sum2/m)
        theta2 = theta2-alpha*(sum3/m)
    return theta0, theta1, theta2



    
    
if __name__ == '__main__':
    theta0 = 0.0
    theta1 = 0.0
    theta2 = 0.0
    iterations = 1500
    alpha = 0.01
#=======================part1:Loading Data============================================
    print('Loading Data...')
    X1, X2, Y = load()
    m = len(Y)
    # Y = load('output_3_1.txt')
    print('m =',m)
    print('X1 =',X1[:10])
    print('X2 =',X2[:10])
    print('Y =',Y[:10])
    
#========================part2:Testing CostFuntion====================================
    print('Testing CostFuntion ... ')
    J = ComputeCost(X1, X2, Y, theta0, theta1, theta2)
    print('With theta0 = 0 and theta1 = 0')
    print('Cost computed =',J)
    # print('Expected cost value (approx) 32.07')
    J = ComputeCost(X1, X2, Y, -1.0, 2.0, 1.0)
    # print('With theta0 = -1 and theta1 = 2')
    print('Cost computed =',J)
    # print('Expected cost value (approx) 54.24')
#========================part3:Gradient descent======================================
    print('Running Gradient Descent ... ')
    theta0, theta1, theta2 = gradientDescent(X1, X2, Y, theta0, theta1, theta2, alpha, iterations)
    print('Theta found by gradient descent:')
    print('theta0 =, theta1 =, theta2 =' ,theta0, theta1, theta2)
    print('Expected theta values (approx)')
    print('thata0 = 0.819236, theta1 = 0.980213')





    
    
    
    
    
    
    