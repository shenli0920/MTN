#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 12:54:34 2022

@author: shenli
"""
import torch
from transformers import BertModel, BertConfig, DNATokenizer

class Nucleic_Featurization():
    def __init__(self):
        pass
    
    def seq2kmer(self,seq,k):
#Convert sequence to k-mers
        kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
        kmers = " ".join(kmer)
        return kmers

    def DNABert(self,kmer,model,tokenizer,config,dir_to_pretrained_model):
        model_input = tokenizer.encode_plus(kmer, add_special_tokens=True, max_length=512)["input_ids"]
        model_input = torch.tensor(model_input, dtype=torch.long)
        model_input = model_input.unsqueeze(0)
        output = model(model_input)
        return output[1]
    
    def process(self,seqs):
        dir_to_pretrained_model = "./DNABERT/examples/model/"
        config = BertConfig.from_pretrained('https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-3/config.json')
        tokenizer = DNATokenizer.from_pretrained('dna3')
        model = BertModel.from_pretrained(dir_to_pretrained_model, config=config)
        embeddings=[]
        for seq in seqs:
            embeddings.append(self.DNABert(self.seq2kmer(seq,3),model,tokenizer,config,dir_to_pretrained_model))   
        return np.concatenate(([emb_vec for emb_vec in embeddings]),axis=1)
