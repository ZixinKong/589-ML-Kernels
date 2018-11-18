#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import Ridge

def krrs(train_x, train_y, test_x, i):
    N = len(train_x)

    lamda = 0.1
    poly_i = i
    K = np.zeros((N, N)) # 200*200
    I = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            K[i][j] = (1 + train_x[i] * train_x[j]) ** poly_i
            if(i == j):
                I[i][j] = 1
                
    alpha = np.dot(np.linalg.inv(K + lamda * I), train_y)

    predicted_y = []
    for n in range(N):
        y = 0
        for i in range(N):
            k = (1 + test_x[n] * train_x[i]) ** poly_i
            y += alpha[i] * k
        predicted_y.append(y)
    return predicted_y

def berr(train_x, train_y, test_x, i):
    N = len(train_x)
    train_x_expansion = np.zeros((N, i+1))

    for n in range(N):
        phi = np.zeros(i+1)
        for j in range(i+1):
            phi[j] = train_x[n] ** j
        train_x_expansion[n] = phi

    test_x_expansion = np.zeros((N, i+1))

    for n in range(N):
        phi = np.zeros(i+1)
        for j in range(i+1):
            phi[j] = test_x[n] ** j
        test_x_expansion[n] = phi

    ridge = Ridge(alpha=0.1)
    ridge.fit(train_x_expansion, train_y)
    prediction = ridge.predict(test_x_expansion)
    return prediction