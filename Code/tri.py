#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import Ridge

def krrs(train_x, train_y, test_x, i):
    N = len(train_x)

    lamda = 0.1
    sigma = 0.5
    poly_i = i
    K = np.zeros((N, N)) # 200*200
    I = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            temp = 0 
            for k in range(1, poly_i+1):
                temp += np.sin(k * sigma * train_x[i]) * np.sin(k * sigma * train_x[j]) + np.cos(k * sigma * train_x[i]) * np.cos(k * sigma * train_x[j])
                K[i][j] = 1 + temp
            if(i == j):
                I[i][j] = 1
                
    alpha = np.dot(np.linalg.inv(K + lamda * I), train_y)

    predicted_y = []
    for n in range(N):
        y = 0
        for i in range(N):
            temp = 0 
            for k in range(1, poly_i+1):
                temp += np.sin(k * sigma * test_x[n]) * np.sin(k * sigma * train_x[i]) + np.cos(k * sigma * test_x[n]) * np.cos(k * sigma * train_x[i])
                k = 1 + temp
            y += alpha[i] * k
        predicted_y.append(y)
    return predicted_y


def berr(train_x, train_y, test_x, i):
    sigma = 0.5
    N = len(train_x)
    train_x_expansion = np.zeros((N, 2*i+1))

    for n in range(N):
        phi = np.zeros(2*i+1)
        phi[0] = 1
        for j in range(1, i+1):
            phi[2*j-1] = np.sin(j * sigma * train_x[n])
            phi[2*j] = np.cos(j * sigma * train_x[n])
        train_x_expansion[n] = phi

    test_x_expansion = np.zeros((N, 2*i+1))

    for n in range(N):
        phi = np.zeros(2*i+1)
        phi[0] = 1
        for j in range(1, i+1):
            phi[2*j-1] = np.sin(j * sigma * test_x[n])
            phi[2*j] = np.cos(j * sigma * test_x[n])
        test_x_expansion[n] = phi
        

    ridge = Ridge(alpha=0.1)
    ridge.fit(train_x_expansion, train_y)
    prediction = ridge.predict(test_x_expansion)
    return prediction