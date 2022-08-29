#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 12:23:43 2022

@author: shenli
"""

import pandas as pd
import torch
import esm
import csv
import numpy as np

class Protein_Featurazation():
    def __init__(self):
        pass

    def ESM(self,seq,model,alphabet,batch_converter):
#ESM only allows a maximum sequence length of 1022 (1024 tokens). We truncate sequences that break the limit.    
        data= [(0,seq[:1022])]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        sequence_representations = []
        for i, (_, seq) in enumerate(data):
            sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))
    
        return sequence_representations[0]


    def process(self,seqs):
#load ESM-1b model    
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        batch_converter = alphabet.get_batch_converter()
# extract protein embeddings.    
        embeddings=[]
        for seq in seqs:
            embeddings.append(self.ESM(seq,model,alphabet,batch_converter).numpy().reshape((1280,1)))
        
        return np.concatenate(([emb_vec for emb_vec in embeddings]),axis=1)
    

