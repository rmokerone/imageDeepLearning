#!/usr/bin/env python
# coding=utf-8
import math
import random

def newfile(file1,filename):
    with open(filename,'w') as g:
        with open(file1,'r') as f:
            for l in f.readlines():
                L=l.strip().split(' ')
                S=int(L[0])+int(L[1])
                g.write(L[0]+' '+L[1]+' '+str(S)+'\n')

def load(filename):
    ret_X1 = []
    ret_X2 = []
    ret_y = []
    with open(filename,'r') as f:
        for l in f.readlines():
            L=l.strip().split(' ')
            L0=list(map(float,list(L)))
            ret_X1.append(L0[0]/800)
            ret_X2.append(L0[1]/800)
            ret_y.append(L0[2]/800)
    return ret_X1, ret_X2, ret_y

def computeCost(X1, X2, y, theta0, theta1, theta2):
    J = 0
    m=len(y)
    for i in range(len(y)):
	    a=theta0+theta1*X1[i]+theta2*X2[i]-y[i]
	    J=J+pow(a,2)
    J=J/(2*m)
    return J

def gradientDescent(X1, X2, y, theta0, theta1, theta2, alpha, num_iters):
    m = len(y)
    for i in range(num_iters):
        J0=0
        J1=0
        J2=0
        for j in range(m):
            a=theta0+theta1*X1[j]+theta2*X2[j]-y[j]
            J0=J0+a
            J1=J1+a*X1[j]
            J2=J2+a*X2[j]
        ta0=theta0-alpha*J0/m
        ta1=theta1-alpha*J1/m
        ta2=theta2-alpha*J2/m
        theta0=ta0
        theta1=ta1
        theta2=ta2
    return theta0, theta1, theta2, i+1, alpha


if __name__ == '__main__':
    # initialization parameter
    theta0 = 0.0
    theta1 = 0.0
    theta2 = 0.0
    iterations = 267
    alpha = 1.02
    #newfile('C:\\work\dataset_1_1.txt','C:\\work\dataset3.txt')#文件已生成
    # ================ Part 1: Loading Data =========================
    print('Loading Data ...')
    X1, X2, y = load('C:\\work\dataset3.txt')
    m = len(y)
    print('m = %s' % m)
    print('X1[:10] = %s' % X1[:10])
    print('X2[:10] = %s' % X2[:10])
    print('y[:10] = %s' % y[:10])
    '''
    # ================ Part 2: Testing CostFuntion ==================
    print('Testing CostFuntion ...')
    # compute initial cost
    J = computeCost(X1, X2, y, theta0, theta1, theta2)
    print('With theta0 = 0 and theta1 = 0 and theta2 = 0')
    print('Cost computed = %f' % J)
    '''
    # ================ Part 3: Gradient descent =====================
    print('Running Gradient Descent ...')
    theta0, theta1, theta2, i, alpha = gradientDescent(X1, X2, y, theta0, theta1, theta2, alpha, iterations)
    print('Theta found by gradient descent:')
    print('theta0 = %.6f, theta1 = %.6f, theta2 = %.6f, i=%d, alpha=%.3f' % (theta0, theta1, theta2, i, alpha))