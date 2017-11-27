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
    with open(filename, 'r') as f:
        X = np.loadtxt(f)
        y = np.sum(X, axis=1, keepdims=True)
        return X, y

""" Mean normailization
参数: 
    vals -- 要归一化的矩阵
返回:
    vals_norm -- 归一化后的矩阵
"""
def featrueScaling(vals):
    vals_mean = vals.mean(axis=0, keepdims=True)
    vals_std = vals.std(axis=0, keepdims=True)
    vals_norm = (vals - vals_mean) / vals_std
    return vals_norm

""" 计算模型的损失
参数:
    X -- 样本X
    y -- 样本y
    theta -- 参数
返回:
    J[0] -- 计算得到的损失，为float类型
"""
def computeCost(X, y, theta):
    m = X.shape[0]
    y_hat = np.dot(X, theta)
    error = y_hat - y
    J = np.dot(error.T, error) / (2 * m)
    return J[0]

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
        y_hat = np.dot(X, theta)
        theta -= alpha * np.dot(X.T, (y_hat - y)) / m
        cost = computeCost(X, y, theta)[0]
        J_log.append(cost)
        if (i+1) % 100 == 0:
            print('iter{%d} cost = %f' % (i+1, cost))
    return theta, J_log

""" 损失显示函数
参数:
    J_log -- 迭代损失的记录数组
返回:
    None
"""
def plotCost(J_log):
    plt.plot(J_log, 'r')
    plt.xlabel("iters num")
    plt.ylabel("J")
    plt.title("J of iters num")
    plt.legend(["cost curve"])
    plt.show()

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
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    y_std = y.std(axis=0, keepdims=True)
    X_test_norm = (X_test - X_mean) / X_std
    X_test_norm = np.hstack((np.ones((2, 1)), X_test_norm))
    predict = np.dot(X_test_norm, theta)*y_std + y_mean
    return predict




if __name__ == '__main__':
    theta = np.zeros((3, 1))
    alpha = 0.1
    # ================ Part 1: Loading Data ==================
    print('Loading Data ...')
    X, y = load('dataset_1_1.txt')
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
    X_norm = np.hstack((np.ones((m, 1)), X_norm))
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
