


#!/usr/bin/env python
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

""" 从文本中载入数据集
参数:
    filename -- 文本文件名
返回:
    X, y -- 样本列表
"""
def load(filename):
    X = np.zeros((1, 1))
    y = np.zeros((1, 1))
    # ================== YOUR CODE HERE =============================
    with open('C:\\Users\Lqj\Desktop\jiqixuexi(jiang)\lecture4\dataset_4.txt', 'r') as f:
    # output = open('output_1_1.txt', 'w')
        data = f.readlines()
        i = 0
        m = len(data)
        X_1 = []
        X_2 = []
        while i<m:
            data1 = data[i].strip().split()
            #print("the type of data1", type(data1))
            i = i+1    
            X_1.append(float(data1[0]))
            X_2.append(float(data1[1]))
        X_1 = np.array(X_1).reshape((-1, 1))
        print('X_1 = ',X_1)
        X_2 = np.array(X_2).reshape((-1, 1))
        print("X_1.shape = ", X_1.shape)
        print("X_2.shape = ", X_2.shape)
        X = np.hstack((X_1,X_2))
        print('X.shape=',X.shape)
        y = np.sum(X,axis=1).reshape(-1,1)
        # print(y)
        # print('y.shape=',y.shape)
    # ===============================================================
    return X, y

""" Mean normailization
参数: 
    vals -- 要归一化的矩阵
返回:
    vals_norm -- 归一化后的矩阵
"""
def featrueScaling(vals):
    vals_norm = np.zeros((1,1))
    # ================== YOUR CODE HERE =============================
    vals1 = np.mean(vals,axis=0)
    vals = vals - vals1 
    vals2 = np.std(vals,axis = 0)
    vals_norm = vals/vals2

    # ===============================================================
    return vals_norm

""" 计算模型的损失
参数:
    X -- 样本X
    y -- 样本y
    theta -- 参数
返回:
    J -- 计算得到的损失，为float类型
"""
def computeCost(X, y, theta):
    m = X.shape[0]
    J = 0.0
    # ================== YOUR CODE HERE =============================
    # m =np.shape(X)
    print('m = ',m)
    print('X.shape= ',X.shape)
    h = np.dot(X,theta)
    J = np.sum(((h - y)**2)/(2*m))


    # ===============================================================
    return J

""" 梯度下降算法
参数:
    X -- 样本X
    y -- 样本y
    theta -- 参数
    alpha -- 学习率
    num_iters -- 迭代次数
返回:
    theta -- 经过num_iters迭代后得到的参数
    J_log -- 每次迭代损失的记录数组，为list类型
"""
def gradientDescent(X, y, theta, alpha, num_iters):
    J_log = []
    m = X.shape[0]
    for i in range(num_iters):
    # ================== YOUR CODE HERE =============================
        h = np.dot(X,theta)
        theta = theta - (alpha*np.dot(X.T,(h - y)))/m
        cost = computeCost(X,y,theta)
        J_log.append(cost)

    # ===============================================================
    return theta, J_log

""" 损失显示函数
参数:
    J_log -- 迭代损失的记录数组
返回:
    None
"""
def plotCost(J_log):
    # ================== YOUR CODE HERE =============================
    plt.plot(J_log)
    plt.title('J of iters num')
    plt.xlabel('iters num')
    plt.ylabel('J')
    plt.legend(['cost curve'])
    plt.show()

    # ===============================================================

""" 数值预测
参数:
    X -- 样本X
    y -- 样本y
    X_test -- 测试集X
    theta -- 参数
返回:
    predict -- 预测结果
"""
def predictVals(X, y, X_test, theta):
    predict = 0
    X_test1 = []
    X_test2 = []
    # ================== YOUR CODE HERE =============================

    X_test1 = np.mean(X,axis = 0)
    X_test = X_test - X_test1
    X_test2 = np.std(X,axis = 0)
    X_test = X_test/X_test2
    m = X_test.shape[0]
    X_norm = np.hstack((np.ones((m, 1)), X_test))
    y_norm = np.dot(X_norm,theta)
    predict = y_norm*np.std(y,axis = 0)+np.mean(y,axis = 0)

    
    # ===============================================================
    return predict




if __name__ == '__main__':
    theta = np.zeros((3, 1))
    alpha = 0.1
    # ================ Part 1: Loading Data ==================
    print('Loading Data ...')
    X, y = load('dataset_4.txt')
    print('The shape of the variable X is: ' , X.shape)
    print("Expected shape of the variable X is (1024, 2)")
    print('The shape of the variable y is: ' , y.shape)
    print("Expected shape of the variable y is (1024, 1)")
    print('X[0] = %s' % X[0])
    print("Expected X[0] = [ 941.  136.]")
    print('y[0] = %s' % y[0])
    print("Expected Y[0] = [ 1077.]")
    
    # ================ Part 2: Feature Scaling ===============
    print('Feature Scaling ...')
    X_norm = featrueScaling(X)
    y_norm = featrueScaling(y)
    print(X_norm.shape)
    print(y_norm.shape)
    print('X_norm[0] = %s' % X_norm[0])
    print("Expected X_norm[0] = [ 1.45964021 -1.3017674 ]")
    print('y_norm[0] = %s' % y_norm[0])
    print("Expected y_norm[0] = [ 0.12507497]")
    # 给X_norm添加第一列全1
    m = X.shape[0]
    X_norm = np.hstack((np.ones((m, 1)), X_norm))#将矩阵在列方向上合并
    # ================ Part 3: Testing costFunction ==========
    print("Testing costFunction ...")
    J = computeCost(X_norm, y_norm, theta)
    print("With theta = ")
    print(theta)
    print("Cost computed = %f" % J)
    theta_for_test = np.array([[1.0], [2.0], [3.0]])
    J = computeCost(X_norm, y_norm, theta_for_test)
    print("Expected cost computed = 0.500000")
    print("With theta = ", )
    print(theta_for_test)
    print("Cost computed = %f" % J)
    print("Expected cost computed = 3.914376")
    # ================= Part 4: Gradient descent ===============
    print("Runing Gradient Descent ...")
    theta, J_log = gradientDescent(X_norm, y_norm , theta, alpha, 1500)
    print("Theta = ")
    print(theta)
    print("Expected Theta approximate= [[ -3.47135278e-18]\
            [  7.16327079e-01]\
            [  7.07119287e-01]]")
    print("type(J_log) = ", type(J_log))
    print("Expected type(J_log) = <class 'list'>")

    # ================= Part 5: Predict values =================
    print("Predict values")
    X_test = np.array([[12.0, 21.0], [123.0, 321.0]])
    predict = predictVals(X, y, X_test, theta)
    print('predict = ', predict)
    print("Expected predict approximate= [[  33.]\
            [ 444.]]")

    # ================= Part 6: Ploting J_log ==================
    print("Ploting J_log ...")
    plotCost(J_log)
