#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from models import MLPmodel
from bayes_opt import BayesianOptimization

class optimization():
    def __init__(self,X_train,y_train,X_test,y_test,task='classification'):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test       
        self.task=task
        self.best_score = 0
    
    def MLP_estimator(self,N_layers,N_neuron,batch_size,max_iter,learning_rate,alpha):

############
  #network size: neurons(16,32,64,128,256,512,1024); layers(1,2,3,4,5,6) 
        N_neuron = 8*(2**int(N_neuron))
    
        temp_hls=[]
        for i in range((int(N_layers))):
            temp_hls.append(int(N_neuron))
        hidden_layer_sizes=tuple(temp_hls)
############    

        #batch_size = 16*(2**int(batch_size))##32,64,128,256 
        batch_size = int(batch_size) ##1,2,4,8; only used for sample data 
    
        max_iter = 5*(2**int(max_iter))##10,20,40,80,160,320
        
        learning_rate = 0.1**(int(learning_rate)+1)#0.01,0.001,0.0001,0.00001
    
        alpha = 0.1**(int(alpha)+1)#0.01,0.001,0.0001,0.00001
    
        model = MLPmodel(hidden_layer_sizes=hidden_layer_sizes,batch_size=batch_size,max_iter=max_iter,learning_rate=learning_rate,alpha=alpha,task=self.task)
        
        score = model.training(self.X_train,self.y_train,self.X_test,self.y_test)
        
        if score>self.best_score:
            self.best_score = score
        return score

    
    def fit(self):
        ann_bo = BayesianOptimization(
            self.MLP_estimator,
            {'N_layers':(1,6.99),
             'N_neuron':(3,9.99),
             'batch_size':(1,4.99),
             'learning_rate':(1,4.99),
             'max_iter':(1,6.99),
             'alpha':(1,4.99)
             })
        ann_bo.maximize()    
