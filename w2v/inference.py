#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   inference.py
@Time    :   2024/07/16 15:04:10
@Author  :   yangqinglin
@Version :   v1.0
@Email   :   yangql1@wedoctor.com
@Desc    :   None
'''
import os
import sys
project_path = os.path.abspath(".")
sys.path.insert(0, project_path)
from BPE.tokenization import YTokenzier
import gensim

class Embed:
    def __init__(self, token_dir, w2v_model) -> None:
        self.w2v_model = w2v_model
        self.tokenizer = YTokenzier.from_pretrained(token_dir)

    def Embed(self,text):
        tokens = self.tokenizer._tokenize(text)
        model = gensim.models.Word2Vec.load(self.w2v_model)
        dic = model.wv.index_to_key
        embed = model.wv[tokens]
        similar = model.wv.most_similar(tokens, topn=10)
        similar_list = [self.tokenizer.convert_tokens_to_string(t[0]) for t in similar]
        return embed, similar_list

if __name__ == '__main__':
    token_dir ="/home/mai-llm-train-service/yql/AI/BPE/toks"
    w2v_model = "/home/mai-llm-train-service/yql/AI/w2v/word2vec.model"
    embeding = Embed(token_dir, w2v_model)
    embed,sim = embeding.Embed("带着走大环和西藏，只用了一次就掉了一地的毛，只能说证明他真的是毛")
    print(embed.shape)