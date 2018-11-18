#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

############################################################################
# implement 5-fold-cross-validation
def credit_card(train_x, train_y):
    result ={}
    
    ##### RBF
    clf = KernelRidge(alpha=1, kernel='rbf', gamma=None)
    MSE = cross_val_score(clf, train_x, train_y, cv=10, scoring='neg_mean_squared_error')
    result["RBF,alpha=1,gamma=None"] = np.mean(0-MSE)
    
    clf = KernelRidge(alpha=1, kernel='rbf', gamma=1)
    MSE = cross_val_score(clf, train_x, train_y, cv=10, scoring='neg_mean_squared_error')
    result["RBF,alpha=1,gamma=1"] = np.mean(0-MSE)
    
    clf = KernelRidge(alpha=1, kernel='rbf', gamma=0.001)
    MSE = cross_val_score(clf, train_x, train_y, cv=10, scoring='neg_mean_squared_error')
    result["RBF,alpha=1,gamma=0.001"] = np.mean(0-MSE)
    
    clf = KernelRidge(alpha=0.0001, kernel='rbf', gamma=None)
    MSE = cross_val_score(clf, train_x, train_y, cv=10, scoring='neg_mean_squared_error')
    result["RBF,alpha=0.0001,gamma=None"] = np.mean(0-MSE)
    
    clf = KernelRidge(alpha=0.0001, kernel='rbf', gamma=1)
    MSE = cross_val_score(clf, train_x, train_y, cv=10, scoring='neg_mean_squared_error')
    result["RBF,alpha=0.0001,gamma=1"] = np.mean(0-MSE)
    
    clf = KernelRidge(alpha=0.0001, kernel='rbf', gamma=0.001)
    MSE = cross_val_score(clf, train_x, train_y, cv=10, scoring='neg_mean_squared_error')
    result["RBF,alpha=0.0001,gamma=0.001"] = np.mean(0-MSE)
    
    ###### Polynomial degree 3
    clf = KernelRidge(alpha=1, kernel='poly', gamma=None, degree=3)
    MSE = cross_val_score(clf, train_x, train_y, cv=10, scoring='neg_mean_squared_error')
    result["Poly,alpha=1,gamma=None"] = np.mean(0-MSE)
    
    clf = KernelRidge(alpha=1, kernel='poly', gamma=1, degree=3)
    MSE = cross_val_score(clf, train_x, train_y, cv=10, scoring='neg_mean_squared_error')
    result["Poly,alpha=1,gamma=1"] = np.mean(0-MSE)
    
    clf = KernelRidge(alpha=1, kernel='poly', gamma=0.001, degree=3)
    MSE = cross_val_score(clf, train_x, train_y, cv=10, scoring='neg_mean_squared_error')
    result["Poly,alpha=1,gamma=0.001"] = np.mean(0-MSE)
    
    clf = KernelRidge(alpha=0.0001, kernel='poly', gamma=None, degree=3)
    MSE = cross_val_score(clf, train_x, train_y, cv=10, scoring='neg_mean_squared_error')
    result["Poly,alpha=0.0001,gamma=None"] = np.mean(0-MSE)
    
    clf = KernelRidge(alpha=0.0001, kernel='poly', gamma=1, degree=3)
    MSE = cross_val_score(clf, train_x, train_y, cv=10, scoring='neg_mean_squared_error')
    result["Poly,alpha=0.0001,gamma=1"] = np.mean(0-MSE)
    
    clf = KernelRidge(alpha=0.0001, kernel='poly', gamma=0.001, degree=3)
    MSE = cross_val_score(clf, train_x, train_y, cv=10, scoring='neg_mean_squared_error')
    result["Poly,alpha=0.0001,gamma=0.001"] = np.mean(0-MSE)
    
    ##### Linear
    clf = KernelRidge(alpha=1, kernel='linear', gamma=None)
    MSE = cross_val_score(clf, train_x, train_y, cv=10, scoring='neg_mean_squared_error')
    result["Linear,alpha=1,gamma=None"] = np.mean(0-MSE)
    
    
    clf = KernelRidge(alpha=0.0001, kernel='linear', gamma=None)
    MSE = cross_val_score(clf, train_x, train_y, cv=10, scoring='neg_mean_squared_error')
    result["Linear,alpha=0.0001,gamma=None"] = np.mean(0-MSE)
    
    return result

###########################################################################
# implement 5-fold-cross-validation
def tumor(train_x, train_y):
    result ={}
    
    ##### RBF
    clf = SVC(C=1.0, kernel='rbf', gamma=1)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["RBF,C=1,gamma=1"] = np.mean(acc)
    
    clf = SVC(C=1.0, kernel='rbf', gamma=0.01)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["RBF,C=1,gamma=0.01"] = np.mean(acc)
    
    clf = SVC(C=1.0, kernel='rbf', gamma=0.001)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["RBF,C=1,gamma=0.001"] = np.mean(acc)
    
    clf = SVC(C=0.01, kernel='rbf', gamma=1)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["RBF,C=0.01,gamma=1"] = np.mean(acc)
    
    clf = SVC(C=0.01, kernel='rbf', gamma=0.01)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["RBF,C=0.01,gamma=0.01"] = np.mean(acc)
    
    clf = SVC(C=0.01, kernel='rbf', gamma=0.001)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["RBF,C=0.01,gamma=0.001"] = np.mean(acc)
    
    clf = SVC(C=0.0001, kernel='rbf', gamma=1)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["RBF,C=0.0001,gamma=1"] = np.mean(acc)
    
    clf = SVC(C=0.0001, kernel='rbf', gamma=0.01)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["RBF,C=0.0001,gamma=0.01"] = np.mean(acc)
    
    clf = SVC(C=0.0001, kernel='rbf', gamma=0.001)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["RBF,C=0.0001,gamma=0.001"] = np.mean(acc)
    
    
    ###### Polynomial degree 3
    clf = SVC(C=1.0, kernel='poly', degree=3, gamma=1)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["Poly.degree=3,C=1,gamma=1"] = np.mean(acc)
    
    clf = SVC(C=1.0, kernel='poly', degree=3, gamma=0.01)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["Poly.degree=3,C=1,gamma=0.01"] = np.mean(acc)
    
    clf = SVC(C=1.0, kernel='poly', degree=3, gamma=0.001)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["Poly.degree=3,C=1,gamma=0.001"] = np.mean(acc)
    
    clf = SVC(C=0.01, kernel='poly', degree=3, gamma=1)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["Poly.degree=3,C=0.01,gamma=1"] = np.mean(acc)
    
    clf = SVC(C=0.01, kernel='poly', degree=3, gamma=0.01)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["Poly.degree=3,C=0.01,gamma=0.01"] = np.mean(acc)
    
    clf = SVC(C=0.01, kernel='poly', degree=3, gamma=0.001)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["Poly.degree=3,C=0.01,gamma=0.001"] = np.mean(acc)
    
    clf = SVC(C=0.0001, kernel='poly', degree=3, gamma=1)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["Poly.degree=3,C=0.0001,gamma=1"] = np.mean(acc)
    
    clf = SVC(C=0.0001, kernel='poly', degree=3, gamma=0.01)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["Poly.degree=3,C=0.0001,gamma=0.01"] = np.mean(acc)
    
    clf = SVC(C=0.0001, kernel='poly', degree=3, gamma=0.001)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["Poly.degree=3,C=0.0001,gamma=0.001"] = np.mean(acc)
    
    
    ###### Polynomial degree 5
    clf = SVC(C=1.0, kernel='poly', degree=5, gamma=1)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["Poly.degree=5,C=1,gamma=1"] = np.mean(acc)
    
    clf = SVC(C=1.0, kernel='poly', degree=5, gamma=0.01)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["Poly.degree=5,C=1,gamma=0.01"] = np.mean(acc)
    
    clf = SVC(C=1.0, kernel='poly', degree=5, gamma=0.001)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["Poly.degree=5,C=1,gamma=0.001"] = np.mean(acc)
    
    clf = SVC(C=0.01, kernel='poly', degree=5, gamma=1)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["Poly.degree=5,C=0.01,gamma=1"] = np.mean(acc)
    
    clf = SVC(C=0.01, kernel='poly', degree=5, gamma=0.01)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["Poly.degree=5,C=0.01,gamma=0.01"] = np.mean(acc)
    
    clf = SVC(C=0.01, kernel='poly', degree=5, gamma=0.001)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["Poly.degree=5,C=0.01,gamma=0.001"] = np.mean(acc)
    
    clf = SVC(C=0.0001, kernel='poly', degree=5, gamma=1)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["Poly.degree=5,C=0.0001,gamma=1"] = np.mean(acc)
    
    clf = SVC(C=0.0001, kernel='poly', degree=5, gamma=0.01)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["Poly.degree=5,C=0.0001,gamma=0.01"] = np.mean(acc)
    
    clf = SVC(C=0.0001, kernel='poly', degree=5, gamma=0.001)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["Poly.degree=5,C=0.0001,gamma=0.001"] = np.mean(acc)
    
    
    ####### Linear
    clf = SVC(C=1.0, kernel='linear', degree=5, gamma=1)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["Linear,C=1,gamma=1"] = np.mean(acc)
    
    clf = SVC(C=0.01, kernel='linear', degree=5, gamma=1)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["Linear,C=0.01,gamma=1"] = np.mean(acc)
    
    clf = SVC(C=0.0001, kernel='linear', degree=5, gamma=1)
    acc = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    result["Linear,C=0.0001,gamma=1"] = np.mean(acc)
    
    return result