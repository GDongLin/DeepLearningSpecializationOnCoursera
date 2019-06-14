#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/14 19:03
# @Author  : GDongLin
# @FileName: __init__.py.py


"""
Welcome to your first (required) programming assignment! You will build a logistic regression classifier to recognize cats. This assignment will step you through how to do this with a Neural Network mindset, and so will also hone your intuitions about deep learning.

Instructions:

Do not use loops (for/while) in your code, unless the instructions explicitly ask you to do so.
You will learn to:

    Build the general architecture of a learning algorithm, including:
        Initializing parameters
        Calculating the cost function and its gradient
        Using an optimization algorithm (gradient descent)
    Gather all three functions above into a main model function, in the right order.
"""

import numpy as np
import matplotlib

matplotlib.use('PS')
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from CourseOneNeuralNetworksAndDeepLearning.WeekTwoLogisticRegressionWithANeuralNetworkMindset.lr_utils import \
    load_dataset


def dataset():
    # 加载原始数据
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    # 将训练样本和测试样本转换成 (num_px  ∗∗  num_px  ∗∗ 3, m), m代表样本数
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    train_set_x_flatten = train_set_x_flatten / 255.
    test_set_x_flatten = test_set_x_flatten / 255.
    data = {
        "train_set_x_flatten": train_set_x_flatten,
        "train_set_y": train_set_y,
        "test_set_x_flatten": test_set_x_flatten,
        "test_set_y": test_set_y,
        "classes": classes
    }
    return data


def linear_cal(w, b, x):
    return np.dot(w.T, x) + b


def sigmoid(z):
    """
    定义sigmoid激活函数
    :param z: 常量或任何shape的numpy数组
    :return: s = sigmoid(z)
    """
    return 1. / (1. + np.exp(-z))


def initialize_with_zeros(dim):
    """
    初始化参数, 直接零初始化, 对, 就是这么简单粗暴
    :param dim: 需要初始化参数维度
    :return:
    """
    w = np.zeros((dim, 1))
    b = 0.
    return w, b


def propagate(w, b, X, Y):
    """
    计算梯度值和代价
    **Hints**:

    Forward Propagation:
    - You get X
    - You compute $A = \sigma(w^T X + b) = (a^{(0)}, a^{(1)}, ..., a^{(m-1)}, a^{(m)})$
    - You calculate the cost function: $J = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)})$

    Here are the two formulas you will be using:

    $$ \frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T\tag{7}$$
    $$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})\tag{8}$$

    :param w: 权重
    :param b: bias
    :param X: 训练样本
    :param Y: 真实标签值
    :return:
        cost: 逻辑回归损失值
        dw: 参数w的梯度值
        db: 参加b的梯度值
    """
    m = X.shape[1]

    Z = linear_cal(w, b, X)
    A = sigmoid(Z)
    assert A.shape == (1, m)

    # 计算cost
    cost = -1. / m * np.sum(Y * np.log(A) + (1. - Y) * np.log(1. - A))
    cost = np.squeeze(cost)

    dw = 1. / m * np.dot(X, (A - Y).T)
    db = 1. / m * np.sum(A - Y)

    grads = {
        "dw": dw,
        "db": db
    }

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    优化函数
    :param w: 权重
    :param b: bias
    :param X: 训练样本
    :param Y: 真实标签值
    :param num_iterations: 迭代次数
    :param learning_rate: 学习速率
    :param print_cost: 是否打印出代价
    :return:
    """
    costs = []
    dw, db = None, None

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        dw = grads["dw"]
        db = grads['db']

        w -= learning_rate * dw
        b -= learning_rate * db

    params = {
        "w": w,
        "b": b
    }
    grads = {
        "dw": dw,
        "db": db
    }
    return params, grads, costs


def predict(w, b, X):
    """
    预测函数
    :param w: 训练后的模型权重
    :param b: 训练后的模型bias
    :param X: 测试样本
    :return:
    """
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    y_hat = sigmoid(linear_cal(w, b, X))
    for i in range(y_hat.shape[1]):
        if y_hat[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1
    assert Y_prediction.shape == (1, m)
    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    W = params["w"]
    B = params["b"]
    Y_prediction_test = predict(W, B, X_test)
    Y_prediction_train = predict(W, B, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


if __name__ == '__main__':
    data = dataset()
    train_set_x = data["train_set_x_flatten"]
    train_set_y = data["train_set_y"]
    test_set_x = data["test_set_x_flatten"]
    test_set_y = data["test_set_y"]
    classes = data["classes"]
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005,
              print_cost=True)
    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()