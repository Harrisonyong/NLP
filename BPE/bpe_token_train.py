#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   bpe_token.py
@Time    :   2024/07/08 16:05:34
@Author  :   yangqinglin
@Version :   v1.0
@Email   :   yangql1@wedoctor.com
@Desc    :   None
'''

import os
from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.processors import TemplateProcessing


class BPETranier():
    def __init__(self, input_file, output_dir, vocab_size=8000):
        self.input_file = input_file
        self.output_dir = output_dir
        self.vocab_size = vocab_size
        
    def __call__(self):
        # 训练BPE模型并保存merges.txt和vocab.json
        
        # 初始化BPE模型和训练器
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        trainer = BpeTrainer(vocab_size=self.vocab_size, min_frequency=2, special_tokens=["<unk>", "<s>", "</s>"])

        # 使用 Unicode 字符预处理器和 ByteLevel 分词器
        tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[
            ("<s>", 1),
            ("</s>", 2),
        ],)
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 训练模型
        tokenizer.train_from_iterator(lines, trainer)
        # 保存模型
        tokenizer.model.save(self.output_dir)
        print(f"Model saved to '{os.path.join(self.output_dir, 'vocab.json')}' and '{os.path.join(self.output_dir, 'merges.txt')}'")


if __name__ == '__main__':
    # 训练
    input_file = "/home/mai-llm-train-service/dataset/classfication/input.txt"
    vocab_size = 12000
    output_dir = "/home/mai-llm-train-service/yql/AI/BPE/toks"
    trainer = BPETranier(input_file=input_file, output_dir=output_dir, vocab_size=vocab_size)
    trainer()

