from bert4keras.snippets import AutoRegressiveDecoder
from bert4keras.tokenizers import Tokenizer
import numpy as np
import os

# 预训练模型的配置（以 BERT 中文预训练模型为例）
config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'

# 加载 Tokenizer
tokenizer = Tokenizer(dict_path, do_lower_case=True)

class SummaryGenerator(AutoRegressiveDecoder):
    def generate(self, text, max_len=50):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_len)
        summary_token_ids = self.random_sample([token_ids, segment_ids])
        summary = tokenizer.decode(summary_token_ids)
        return summary

# 初始化 SummaryGenerator
summary_generator = SummaryGenerator()

# 输入文本
text = "这是一个示例文本，用于演示如何使用 BERT 模型生成指定长度的摘要。"

# 生成摘要
summary = summary_generator.generate(text)
print(summary)
