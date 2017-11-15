#!/usr/bin/env python
# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

class linearRegression(object):
    def __init__(self):
        self.X1 = []
        self.X2 = []
        self.y = []
        self.X1_norm = []
        self.X2_norm = []
        self.y_norm = []
        self.m = 0
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.theta2 = 0.0
        self.alpha = 0.01
        self.log_J = []

    def setAlpha(self, alpha):
        self.alpha = alpha
    
    def setTheta(self, theta0, theta1, theta2):
        self.theta0 = theta0
        self.theta1 = theta1
        self.theta2 = theta2

    def load(self, filename):
        with open(filename, 'r') as f:
            for line in f.readlines():
                x1, x2 = [float(i) for i in line.strip().split()]
                self.X1.append(x1)
                self.X2.append(x2)
                self.y.append(x1 + x2)
            self.m = len(self.X1)

    def featureScaling(self):
        self.x1_mean = sum(self.X1) / self.m
        self.x2_mean = sum(self.X2) / self.m
        self.y_mean = sum(self.y) / self.m
        self.x1_variance = abs(min(self.X1) - max(self.X1))
        self.x2_variance = abs(min(self.X2) - max(self.X2))
        self.y_variance = abs(min(self.y) - max(self.y))
        for x1, x2, y in zip(self.X1, self.X2, self.y):
            self.X1_norm.append((x1 - self.x1_mean) / self.x1_variance)
            self.X2_norm.append((x2 - self.x2_mean) / self.x2_variance)
            self.y_norm.append((y - self.y_mean) / self.y_variance)
        print(self.X1_norm[:10])
        print(self.X2_norm[:10])
        print(self.y_norm[:10])
    
    def featureScalingAverage(self, variance):
        for x1, x2, y in zip(self.X1, self.X2, self.y):
            self.X1_norm.append(x1 / variance)
            self.X2_norm.append(x2 / variance)
            self.y_norm.append(y / variance)
        print(self.X1_norm[:10])
        print(self.X2_norm[:10])
        print(self.y_norm[:10])


    def computeCost(self):
        J = 0
        for x1, x2, y in zip(self.X1_norm, self.X2_norm, self.y_norm):
            predict = self.theta0 + self.theta1*x1 + self.theta2*x2
            J += pow((predict - y), 2.0)
        return J

    def gradientDescent(self, num_iters):
        for i in range(num_iters):
            delta_theta0 = 0.0
            delta_theta1 = 0.0
            delta_theta2 = 0.0
            for x1, x2, y in zip(self.X1_norm, self.X2_norm, self.y_norm):
                predict = self.theta0 + self.theta1*x1 + self.theta2*x2
                delta_theta0 += (predict - y)
                delta_theta1 += (predict - y) * x1
                delta_theta2 += (predict - y) * x2
            self.theta0 -= self.alpha * delta_theta0 / self.m
            self.theta1 -= self.alpha * delta_theta1 / self.m
            self.theta2 -= self.alpha * delta_theta2 / self.m
            cost = self.computeCost()
            if i % 100 == 0:
                print('iter{%d} cost = %.6f' % (i, cost))
            self.log_J.append(cost)
        plt.plot(self.log_J)
        return self.theta0, self.theta1, self.theta2

    def predict(self, x1, x2, isAverage=False):
        if isAverage:
            y = self.theta0 + self.theta1 * x1 + self.theta2 * x2
        else:
            x1_norm = (x1 - self.x1_mean) / self.x1_variance
            x2_norm = (x2 - self.x2_mean) / self.x2_variance
            y_norm = self.theta0 + self.theta1*x1_norm + self.theta2*x2_norm
            y = y_norm * self.y_variance + self.y_mean
        return y    
        

if __name__ == '__main__':
    model = linearRegression()
    model.load('dataset_1_1.txt')
    model.setAlpha(2.0)
    model.featureScaling()
    #model.featureScalingAverage(1024.0)
    theta0, theta1, theta2 = model.gradientDescent(1000)
    print(theta0, theta1, theta2)
    print(model.predict(12, 31))
    plt.show()
