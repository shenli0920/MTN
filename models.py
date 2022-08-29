#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 11:16:59 2022

@author: shenli
"""
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr


def Rp(x,y):
    return pearsonr(x,y)[0]

class MLPmodel():
    def __init__(self,hidden_layer_sizes,batch_size,max_iter,learning_rate,alpha,task='classification'):
        if task == 'classification':
            self.model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                batch_size=batch_size,
                max_iter=max_iter,
                learning_rate_init=learning_rate,
                alpha=alpha
                )
            self.eval = accuracy_score
        elif task == 'regression':
            self.model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                batch_size=batch_size,
                max_iter=max_iter,
                learning_rate_init=learning_rate,
                alpha=alpha
                )
            self.eval = Rp
        
    
    
    def training(self,X_train,y_train,X_test,y_test):
        self.model.fit(X_train,y_train)
        return self.eval(self.model.predict(X_test),y_test)
        
class GBDTmodel():
    def __init__(self,n_estimators,max_depth,min_samples_split,learning_rate,subsample,max_features,task='regression'):
        if task=='classfication':
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                learning_rate=learning_rate,
                subsample=subsample,
                max_features=max_features
                )
            self.eval = accuracy_score
            
        elif task=='regression':
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                learning_rate=learning_rate,
                subsample=subsample,
                max_features=max_features
                )
            self.eval = Rp
            
    def training(self,X_train,y_train,X_test,y_test):
        self.model.fit(X_train,y_train)
        return self.eval(self.model.predict(X_test),y_test)