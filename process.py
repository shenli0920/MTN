#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 16:04:42 2022

@author: shenli
"""
from P import Protein_Featurazation
from N import Nucleic_Featurization
#from L import Ligand_Featurization

class process_seq_pair():
    def __init__(self,data,method='Stack'):
        self.data = data
        self.method= method
        if self.method != 'Stack':
            if self.data.shape[1] != 2 or self.data[0,0] != self.data[0,1]:
                print('Error! '+method+' could only be applied for bi-homotransformers.')
        
    def process_transformers_feature(self):
        self.Transformer_embeddings=[]
        for i in range(self.data.shape[1]):
            seqs = self.data[1:,i]
            if self.data[0,i] == 'Protein':
                processor = Protein_Featurazation()
            elif self.data[0,i] == 'Nucleic Acid':
                processor = Nucleic_Featurazation()
            elif self.data[0,i] == 'Ligand':
                processor = Ligand_Featurazation()
            
            self.Transformer_embeddings.append(processor.process(seqs))
            
    def process_multitransformers_feature(self):
        if self.method == 'Stack':
            self.Feature = np.concatenate(self.Transformer_embeddings,axis=1)
        elif self.method == 'Diff':
            scaler = StandardScaler()
            self.Feature = scaler.fit_transform(np.abs(self.Transformer_embeddings[0]-self.Transformer_embeddings[1]))
        elif self.method == 'Prod':
            scaler = StandardScaler()
            self.Feature = scaler.fit_transform(self.Transformer_embeddings[0]*self.Transformer_embeddings[1])
            
        
            
            