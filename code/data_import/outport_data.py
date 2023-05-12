
import re
def full_width_to_half_width(s):
    new_s = ""
    for char in s:
        inside_code = ord(char)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        new_s += chr(inside_code)
    return new_s

# 过滤文本中的中文字符、标点符号和空格。
def filter_chinese_and_punctuations(text):
    
    # 定义正则表达式，匹配中文、数字和标准的标点符号、英文字符和空格、回车符、制表符等，以及键盘上数字哪一行的所有符号
    pattern = re.compile(r'[\u4e00-\u9fa5\d ，。！？、；：‘’“”（）《》【】\[\]【】a-zA-Z,.!?;:\'"/\\\{\}\(\)\<\>\+\-\*/=~=^`|&#%@_\n\r\t]+')
    # 使用正则表达式过滤文本
    result = pattern.findall(text)
    filtered_text = ''.join(result)
    filtered_text = re.sub(r'u\(b\d{10}N.*?dNl\}', '', filtered_text)
    
    # 删除开头第一个逗号或句号之前的文本（考虑全角和半角）
    filtered_text = re.sub(r'^.*?[，。,.]', '', filtered_text)

    # 删除最后一个逗号或句号之后的文本（考虑全角和半角）
    filtered_text = re.sub(r'[，。,.][^.]*$', '', filtered_text)

    # 将全角符号转换为半角符号
    filtered_text = full_width_to_half_width(filtered_text)
    # 过滤掉大致的特定字符串，不考虑日期
    # 过滤掉连续的点字符
    filtered_text = re.sub(r'\.{3,}', '', filtered_text)
    # 汉字中间的空格给去掉，pdf转译出来有很多空格
    filtered_text = re.sub(r'([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])', r'\1\2', filtered_text)
    filtered_text = re.sub(r'\s+', ' ', filtered_text).strip()
    return filtered_text


from typing import List
import tiktoken
# 将文本分割为最多包含1000个token的块，用来做Qdrant的索引。
def split_text_to_chunks(text: str, max_tokens: int = 800, loop_back_num:int=50) -> List[str]:
    text = filter_chinese_and_punctuations(text)
    encoding = tiktoken.get_encoding("gpt2")
    chunks = []
    current_chunk = ""
    current_chunk_token_count = 0
    index = 0

    while index < len(text):
        char = text[index]
        current_chunk += char
        char_token_count = len(encoding.encode(char))
        current_chunk_token_count += char_token_count

        if current_chunk_token_count >= max_tokens:
            look_back = 0
            split_point = len(current_chunk)
            while look_back < loop_back_num and split_point - look_back > 0:
                if current_chunk[split_point - look_back - 1] in {'\n', ' ', '　', '。', '.', '，',','}:
                    split_point -= look_back
                    break
                look_back += 1

            chunks.append(current_chunk[:split_point])
            current_chunk = current_chunk[split_point:]
            current_chunk_token_count = len(encoding.encode(current_chunk))
            # index -= look_back
        index += 1

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def preprocess_text(text):
    # 移除特殊字符和多余的空白符
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
from qdrant_client.models import Filter, FieldCondition
import numpy as np

# Qdrant client setup
client = QdrantClient(host="localhost", port=6333)  # Replace with your Qdrant host and port

collection_name = "heack"  # Replace with your collection name

# Fetch data from Qdrant
points_to_retrieve = 1000000  # You may need to adjust this number based on your requirements

# Use a random query vector
query_vector = np.random.randn(768).tolist()  # Replace 768 with the actual dimension of your embeddings

points = client.search(collection_name=collection_name, limit=points_to_retrieve, query_vector=query_vector, score_threshold=-10.0) #top=points_to_retrieve,
print(f"{len(points)}")
# Save data to train.json
train_data = []

for point in points:
    payload = point.payload
    text = payload["text"]
    filtered_text = filter_chinese_and_punctuations(text)
    if filtered_text == "":
        continue
    train_data.append(filter_chinese_and_punctuations(text))

with open("data/train.json", "w", encoding="utf-8") as outfile:
    json.dump(train_data, outfile, ensure_ascii=False, indent=4)
