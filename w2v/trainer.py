#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   data_helper.py
@Time    :   2024/06/20 14:34:39
@Author  :   yangqinglin
@Version :   v1.0
@Email   :   yangql1@wedoctor.com
@Desc    :   None
'''
import os
import sys
project_path = os.path.abspath(".")
sys.path.insert(0, project_path)
from gensim.models import word2vec
from BPE.tokenization import YTokenzier

class W2VEmbedTrian():
    def __init__(self, token_dir, input_txt) -> None:
        self.tokenizer = YTokenzier.from_pretrained(token_dir)
        self.input_txt = input_txt
        
    def get_data(self,input_txt):
        self.corpus = []
        with open(input_txt, "r") as f:
            for line in f:
                self.corpus.append(self.tokenizer._tokenize(line))
                
    def train(self):
        self.get_data(self.input_txt)
        print(len(self.corpus))
        model = word2vec.Word2Vec(
            self.corpus,
            vector_size=128,# 词向量纬度
            sg=1,# ski-gram
            window=3, # 滑
            min_count=1, # 最小词频
            workers = 8,
            epochs=2,
        )
        model.wv.save_word2vec_format('data.vector', binary=False)
        model.save('word2vec.model')

    
if __name__ == '__main__':
    token_dir = "/home/mai-llm-train-service/yql/AI/BPE/toks"
    input_txt = "/home/mai-llm-train-service/dataset/classfication/input.txt"
    w2v_corpus = W2VEmbedTrian(token_dir, input_txt)
    w2v_corpus.train()