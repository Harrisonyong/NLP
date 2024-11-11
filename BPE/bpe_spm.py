#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   bpe_spm.py
@Time    :   2024/07/08 14:10:09
@Author  :   yangqinglin
@Version :   v1.0
@Email   :   yangql1@wedoctor.com
@Desc    :   None
'''

import sentencepiece as spm

def train(input_file):
    spm.SentencePieceTrainer.Train(f'--input={input_file} --model_prefix=bpe_sem --vocab_size=12000 --model_type=bpe')

def encode(text, model_file):
    # 加载训练好的模型
    sp = spm.SentencePieceProcessor(model_file=model_file)

    # 对输入文本进行分词
    tokens = sp.EncodeAsPieces(text)
    print("Pieces:", tokens)

    # 将分词结果转换为ID
    token_ids = sp.EncodeAsIds(text)
    print("IDs:", token_ids)

    # 从ID恢复原始文本
    reconstructed_text = sp.decode_ids(token_ids)
    print("Reconstructed text:", reconstructed_text)
    for token, token_id in zip(tokens, token_ids):
        print(f"Token: {token}, ID: {token_id}, In Vocab: {sp.id_to_piece(token_id)}")

if __name__ == '__main__':
    input_file = "/home/mai-llm-train-service/dataset/classfication/input.txt"
    bpe_file = "/home/mai-llm-train-service/yql/AI/BPE/bpe_sem.model"
    encode("你", bpe_file)
    