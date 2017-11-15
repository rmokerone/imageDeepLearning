#!/usr/bin/env python
# coding=utf-8
import os
def load(filename):
    ret_X1=[]
    ret_X2=[]
    ret_y= []
    os.getcwd()
    os.chdir('C:\\Users\Administrator\Desktop\lecture1')
    # ================== YOUR CODE HERE ============================= 
    f=open('dataset_1_1.txt','r')
    f1=open('data.txt','r')
    while True:
        x=f.readline()
        if x=='':
            break
        m=x.split()
        ret_X1.append(float(m[0])/400)
        ret_X2.append(float(m[1])/400)
    while True:
        y=f1.readline()
        if y=='':
            break
        ret_y.append(float(y)/400)
      # ===============================================================
    return ret_X1,ret_X2,ret_y

def computeCost(X1,X2,y,theta0,theta1,theta2):
    J = 0
    # ================== YOUR CODE HERE =============================
    i=0
    d=len(y)
    while i<d:
        J=J+((theta0+theta1*X1[i]+theta2*X2[i]-y[i])**2)/(2.0*d)
        i=i+1          
    # ===============================================================
    return J

def gradientDescent(X1,X2, y, theta0, theta1,theta2 ,alpha, num_iters):
    for i in range(num_iters):
         # ================== YOUR CODE HERE =========================
        p=0
        q=0
        j=0
        r=0
        while j<m:
            p=p+theta0+theta1*X1[j]+theta2*X2[j]-y[j]
            q=q+(theta0+theta1*X1[j]+theta2*X2[j]-y[j])*X1[j]
            r=r+(theta0+theta1*X1[j]+theta2*X2[j]-y[j])*X2[j]
            j=j+1
        theta0=theta0-alpha*1.0/m*p
        theta1=theta1-alpha*1.0/m*q
        theta2=theta2-alpha*1.0/m*r
        # ===========================================================
    return theta0, theta1


if __name__ == '__main__':
    # initialization parameter
    theta0 = 0.0
    theta1 = 0.0
    theta2=0.0
    iterations =1000
    alpha = 0.05
    # ================ Part 1: Loading Data =========================
    print('Loading Data ...')
    X1,X2,y = load('dataset_2_1.txt')
    m = len(y)
    print('m = %s' % m)
    print('X1[:10] = %s' % X1[:10])
    print('x2[:10] = %s' % X2[:10])
    print('y[:10] =%s' %y[:10])
        # ================ Part 3: Gradient descent =====================
    print('Running Gradient Descent ...')
    theta0, theta1= gradientDescent(X1,X2, y, theta0, theta1,theta2, alpha, iterations)
    print('Theta found by gradient descent:')
    print('theta0 = %.6f, theta1 = %.6f' % (theta0,theta1))
     
