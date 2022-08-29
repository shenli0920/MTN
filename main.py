#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 20:36:51 2022

@author: shenli
"""
import pandas as pd
import numpy as np
from process import process_seq_pair
from HyperParaOptimization import optimization
from models import MLPmodel,GBDTmodel
import warnings
warnings.filterwarnings("ignore")

#read data
def read_process_data(filepath):
    seq_train,seq_test,y_train,y_test = pd.read_csv(filepath+'/seq_train.csv',header=None).values[:,:-1],pd.read_csv(filepath+'/seq_test.csv',header=None).values[:,-1],pd.read_csv(filepath+'/seq_train.csv',header=None).values[:,:-1],pd.read_csv(filepath+'/seq_test.csv',header=None).values[:,-1]
#process multitransformers features
    processor = process_seq_pair()
    X_train,X_test = processor.fit(np.concatenate((seq_train,seq_test[1:,:])))[:(seq_train.shape[0]-1),:],processor.fit(np.concatenate((seq_train,seq_test[1:,:])))[(seq_train.shape[0]-1):,:]
    return X_train,y_train,X_test,y_test
#GBDT model: hyperparameters are chosen based on size of datasets.
def GBDT(X_train,y_train,X_test,y_test,task_type):
    #choose hyperpara 
    num=np.shape(X_train)[0]
    if(num<1000):
        i=10000
        j=7
        k=3
        m=2
    elif(num>=1000 and num<10000):
        i=10000
        j=8
        k=4
        m=3
    elif(num>=10000):
        i=70000
        j=5
        k=9
        m=9
    
    model = GBDTmodel(i,j,k,0.001,0.1*m,'sqrt')
    print('GBDT result:',model.training(X_train,y_train,X_test,y_test,task_type))
##MLP model use bayesian optimization to search the best combination of hyperparameters.
def MLP(X_train,y_train,X_test,y_test,task_type):
    opt = optimization(X_train,y_train,X_test,y_test,task_type)
    opt.fit()
    print('MLP result:',opt.best_score)

    
task_type = 'regression' #'classification'
filepath = 'sample_data/PN'

X_train,y_train,X_test,y_test = read_process_data(filepath)
GBDT(X_train,y_train,X_test,y_test,task_type)
MLP(X_train,y_train,X_test,y_test,task_type)
