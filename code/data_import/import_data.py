from qdrant_client import QdrantClient
# import qdrant_openapi_client
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
from qdrant_client.models import Filter, FieldCondition
import os
import tqdm
# import openai
import PyPDF2
import uuid
import unicodedata
import tiktoken
import docx
import docx2txt
import sys

import torch
from text2vec import SentenceModel
from transformers import PegasusForConditionalGeneration

"""
找到重复的文件，并且移动到指定的目录中
"""
import os
import hashlib
import shutil

def file_hash(file_path):
    with open(file_path, 'rb') as f:
        file_data = f.read()
        file_hash = hashlib.md5(file_data).hexdigest()
    return file_hash


def find_duplicate_files_and_move(src_folder, duplicate_folder):
    if not os.path.exists(duplicate_folder):
        os.makedirs(duplicate_folder)

    file_hashes = {}
    for root, _, files in os.walk(src_folder):
        for file in files:
            file_path = os.path.join(root, file)
            current_file_hash = file_hash(file_path)

            if current_file_hash in file_hashes:
                existing_file_path = file_hashes[current_file_hash]
                if len(file) > len(os.path.basename(existing_file_path)):
                    duplicate_file_path = os.path.join(duplicate_folder, file)
                    shutil.move(file_path, duplicate_file_path)
                    print(f"Moved duplicate file: {file_path} to {duplicate_file_path}")
                else:
                    duplicate_file_path = os.path.join(duplicate_folder, os.path.basename(existing_file_path))
                    shutil.move(existing_file_path, duplicate_file_path)
                    print(f"Moved duplicate file: {existing_file_path} to {duplicate_file_path}")
                    file_hashes[current_file_hash] = file_path
            else:
                file_hashes[current_file_hash] = file_path



def filter_chinese_and_punctuations(text):
    # 定义正则表达式，匹配中文、数字和标准的标点符号、英文字符和空格、回车符、制表符等，以及键盘上数字哪一行的所有符号
    pattern = re.compile(r'[\u4e00-\u9fa5\d ，。！？、；：‘’“”（）《》【】\[\]【】a-zA-Z,.!?;:\'"/\\\{\}\(\)\<\>\+\-\*/=~=^`|&#%@_\n\r\t]+')

    # # 定义正则表达式，匹配中文、数字和标准的标点符号、英文字符和空格、回车符、制表符等
    # pattern = re.compile(r'[\u4e00-\u9fa5\d ，。！？、；：‘’“”（）《》【】\[\]【】a-zA-Z,.!?;:()\n\r\t]+')
    # 使用正则表达式过滤文本
    result = pattern.findall(text)
    filtered_text = ''.join(result)
    filtered_text = re.sub(r'([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])', r'\1\2', filtered_text)
    return filtered_text


# from tokenizers_pegasus import PegasusTokenizer

# from transformers import BertTokenizer, BertForMaskedLM

# def get_summary(text, summary_length=512, model_name="bert-base-chinese"):
#     # 加载预训练的 BERT 模型和分词器
#     model = BertForMaskedLM.from_pretrained(model_name)
#     tokenizer = BertTokenizer.from_pretrained(model_name)

#     # 对输入文本进行预处理
#     inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

#     # 生成摘要
#     with torch.no_grad():
#         outputs = model.generate(inputs.input_ids, max_length=summary_length, num_return_sequences=1, no_repeat_ngram_size=2)

#     # 解码生成的摘要并返回结果
#     summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return summary

# from transformers import GPT2LMHeadModel, BertTokenizer, TextGenerationPipeline
# from transformers import pipeline


# def get_summary(text, summary_length=50):
#     summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
#     return summarizer(text, max_length=summary_length, min_length=30, do_sample=False)

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
summary_model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(summary_model_name)
summary_model = AutoModelForSeq2SeqLM.from_pretrained(summary_model_name)
def get_summary(text, summary_length=200):
    input_ids = tokenizer(
        [WHITESPACE_HANDLER(text)],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )["input_ids"]

    output_ids = summary_model.generate(
        input_ids=input_ids,
        max_length=summary_length, #84
        no_repeat_ngram_size=2,
        num_beams=4
    )[0]

    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return summary
    # print(summary)
# def get_summary(text, summary_length=50, model_name="uer/gpt2-chinese-cluecorpussmall"):
#     # 加载预训练的 GPT-2 模型和分词器
#     model = GPT2LMHeadModel.from_pretrained(model_name)
#     tokenizer = BertTokenizer.from_pretrained(model_name)
#     text_generator = TextGenerationPipeline(model, tokenizer)   
#     # 对输入文本进行预处理
#     inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

#     # 生成摘要
#     with torch.no_grad():
#         outputs = model.generate(inputs.input_ids, max_length=summary_length, num_return_sequences=1, no_repeat_ngram_size=2)

#     # 解码生成的摘要并返回结果
#     summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return summary


namespace = uuid.NAMESPACE_URL

# from textrank4zh import TextRank4Sentence

# def get_title(text, max_len=300):
#     tr4s = TextRank4Sentence()
#     tr4s.analyze(text=text, lower=True, source='all_filters')
#     summary_sentences = tr4s.get_key_sentences(num=1)

#     if len(summary_sentences) > 0:
#         summary = summary_sentences[0]['sentence']
#         if len(summary) <= max_len:
#             return summary
#         else:
#             return summary[:max_len] + "..."
#     else:
#         return text[:max_len] + "..." if len(text) > max_len else text

embed_model = SentenceModel('shibing624/text2vec-base-chinese')
def to_embeddings(text):
    embeddings = embed_model.encode([text])
    return embeddings[0].tolist()
    # sentence_embeddings = openai.Embedding.create(
    #     model="text-embedding-ada-002",
    #     input=text
    # )
    # return sentence_embeddings["data"][0]["embedding"]

# text-embedding-ada-002的最大输入是8196 token
# def num_tokens_from_messages(text, model="gpt-3.5-turbo-0301"):
#     """Returns the number of tokens used by a list of messages."""
#     try:
#         encoding = tiktoken.encoding_for_model(model)
#     except KeyError:
#         print("Warning: model not found. Using cl100k_base encoding.")
#         encoding = tiktoken.get_encoding("cl100k_base")
#     if model == "gpt-3.5-turbo":
#         # print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
#         return num_tokens_from_messages(text, model="gpt-3.5-turbo-0301")
#     elif model == "gpt-4":
#         # print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
#         return num_tokens_from_messages(text, model="gpt-4-0314")
#     elif model == "gpt-3.5-turbo-0301":
#         tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
#         tokens_per_name = -1  # if there's a name, the role is omitted
#     elif model == "gpt-4-0314":
#         tokens_per_message = 3
#         tokens_per_name = 1
#     else:
#         raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
#     return len(encoding.encode(text))

from langchain.text_splitter import TokenTextSplitter
def split_text_to_chunks(text, max_tokens=1000):
    text = filter_chinese_and_punctuations(text)
    text = text.replace("\n\n","\n").replace("  "," ").replace("\t\t","\t").replace("..",".")
    text_splitter = TokenTextSplitter(chunk_size=max_tokens, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    return texts

# import re
# def split_text_to_chunks(text, max_tokens=1000):
    
#     text = text.replace("  ","")
#     text = text.replace("\t","")
    
#     paragraphs = re.split(r'[\n。.]', text)
#     chunks = []
#     current_chunk = ""

#     for paragraph in paragraphs:
#         current_length = num_tokens_from_messages(current_chunk)
#         paragraph_length = num_tokens_from_messages(paragraph)

#         if current_length + paragraph_length <= max_tokens:
#             if current_chunk:
#                 current_chunk += "." if paragraph_length > 0 else ""
#             current_chunk += paragraph
#         else:
#             chunks.append(current_chunk)
#             current_chunk = paragraph

#     if current_chunk:
#         chunks.append(current_chunk)

#     return chunks

from PyPDF2.errors import PdfReadError
def get_pdf_chunks(file_path, max_chunk_size=1000):
    chunks = []
    try:
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            num_pages = len(pdf_reader.pages)
            pages = [pdf_reader.pages[i].extract_text() for i in range(num_pages)]

        full_text = "".join(pages)
        chunks = split_text_to_chunks(full_text, max_chunk_size)
    except PdfReadError:
        print(f"Warning: Unable to read the PDF file '{file_path}', EOF marker not found. Skipping this file.")
    return chunks


# if __name__ == '__main__':
#     for root, dirs, files in os.walk("./source_data"):
#         for file in tqdm.tqdm(files):
#             file_path = os.path.join(root, file)
#             file_update_time = os.path.getmtime(file_path)
#             print(f"processing file_path:{file_path}")

#             # print(f"existing_points:{file_path}")
#             file_type = "pdf" # 将来打算再加上text, web, userinput等类型
#             if True:
#                 if file.lower().endswith('.pdf'):
#                     chunks = get_pdf_chunks(file_path)
                    
#                 elif file.lower().endswith('.docx'):
#                     file_type = "word"
#                     doc = docx.Document(file_path)
#                     text = '\n'.join([para.text for para in doc.paragraphs])
#                     chunks = split_text_to_chunks(text)
#                 elif file.lower().endswith('.doc'):
#                     print("cannot process .doc file, please convert it to .docx first, next...")
#                     continue
#                     # file_type = "word"
#                     # text = docx2txt.process(file_path)
#                     # chunks = split_text_to_chunks(text)
#                 else:
#                     file_type = "text"
#                     with open(file_path, 'r', encoding='utf-8') as f:
#                         text = f.read()
#                         chunks = split_text_to_chunks(text)
#                 text = "".join(chunks)
#                 print("---------------------original text---------------------")
#                 print(text)
#                 print("---------------------summary---------------------")
#                 print(get_summary(text))
                      
#                 # print(text, get_summary(text))
#                 # for chunk in chunks:
#                 #     print("-----------------------", chunk)
#                 #     title = get_summary(chunk)
#                 #     print(f"*********** title:{title}")
   
# sys.exit(0)
 
def main_loop(collection_name, base_dir):
    # 清除掉重复的文件
    print("start to delete duplicate files")
    src_folder = base_dir  # 替换为需要遍历的文件夹
    duplicate_folder = "duplicate_delete"
    find_duplicate_files_and_move(src_folder, duplicate_folder)
    
    client = QdrantClient("127.0.0.1", port=6333)
    try:
        # Get information about existing collection
        collection_exists = client.get_collection(collection_name)
        print(f"{collection_exists} exists")
    except Exception as e:
        print(f"Collection {collection_name} not exists, create it")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
    processed_list_file = os.path.join(base_dir, "processed_list")
    try:
        with open(processed_list_file, "r") as f:
            processed_files = set(line.strip() for line in f)
    except FileNotFoundError:
        processed_files = set()  # If the file does not exist, create an empty set

    # count = 0
    for root, dirs, files in os.walk(base_dir):
        for file in tqdm.tqdm(files):
            file_path = os.path.join(root, file)
            file_name = os.path.basename(file_path)
            if file_name == processed_list_file:
                continue
            if file_path in processed_files:
                print(f"Skipping already processed file: {file_path}")
                continue
            file_update_time = os.path.getmtime(file_path)
            file_uuid = str(uuid.uuid5(namespace, f"{file_path}_0")) # check file chunk 0 if exists
                
            print(f"processing file_path:{file_path} with UUiD:{file_uuid} to collection:{collection_name}")
            existing_points = client.retrieve(
                collection_name=collection_name,
                ids=[file_uuid], # check chunk 0 if exists
                with_payload=True
            )
            # 无论如何都添加到这个list，因为要么已经数据库存在了，但是文件里面没有，要么这会就放到数据库了，所以都应该在list文件中
            processed_files.add(file_path)
            with open(processed_list_file, "w") as f:
                for fn in processed_files:
                    f.write(f"{fn}\n")
            """
            按照如下方法进行索引：
            文件的文本分为1000个字符的chunk， 由多个chunk组成
            每个chunk的索引由一下内容组成：
            1. 文件名
            2. 第一页的chunk的summary
            3. 本页的summary
            4. 本页的内容
            然后把第一页也就是第一个chunk做summary，这个summary+文件名和其他所有的chunk每个都结合做索引，但是存储内容只有chunk自身的内容
            """
            # print(f"existing_points:{file_path}")
            file_type = "pdf" # 将来打算再加上text, web, userinput等类型
            if not existing_points:
                
                if file.lower().endswith('.pdf'):
                    chunks = get_pdf_chunks(file_path)
                elif file.lower().endswith('.docx'):
                    file_type = "word"
                    doc = docx.Document(file_path)
                    text = '\n'.join([para.text for para in doc.paragraphs])
                    chunks = split_text_to_chunks(text)
                elif file.lower().endswith('.doc'):
                    print("cannot process .doc file, please convert it to .docx first, next...")
                    continue
                elif file.lower().endswith('.txt'):
                    file_type = "text"
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        chunks = split_text_to_chunks(text)
                else:
                    continue
                if len(chunks) > 0:
                    chunk0_summary = get_summary(chunks[0])
                for file_chunk, chunk_text in enumerate(chunks):
                    chunk_summary = get_summary(chunk_text)
                    point_id = str(uuid.uuid5(namespace, f"{file_path}_{file_chunk}"))
                    embedding_text = f"{file_name} {chunk0_summary} {chunk_summary} {chunk_text}"
                    print("---------------------embedding_text---------------------")
                    print(embedding_text)
                    embedding_vector = to_embeddings(embedding_text)
                    # print(f"embedding_vector:{embedding_vector}")
                    print(f"inserting...... point_id:{point_id} file_name:{file_name}\nchunk_summary:{chunk_summary}\nchunk_text:{chunk_text}\n ")
                    client.upsert(
                        collection_name=collection_name,
                        wait=True,
                        points=[
                            # user file_path as id, for future search
                            PointStruct(id=point_id, vector=embedding_vector, payload={
                                            "title": chunk_summary, 
                                            "text": chunk_text, 
                                            "file_summary": chunk0_summary,
                                            "file_type":file_type, 
                                            "file_update_time": file_update_time, 
                                            "file_name": file_name,
                                            "file_uuid":file_uuid, 
                                            "file_chunk": file_chunk
                                         }),
                        ],
                    )
import time
import json
import argparse

parser = argparse.ArgumentParser(description="Update collections with specified collection name")
parser.add_argument("--collection_name", type=str, required=True, help="Specify the collection name")
args = parser.parse_args()
collection_name = args.collection_name

if __name__ == '__main__':
    with open('../config.json', 'r') as f:
        config = json.load(f)
        
    if collection_name not in config['base_dir']:
        raise KeyError(f"Collection '{collection_name}' is not defined in config file.")
    base_dir = config['base_dir'][collection_name]
    if not os.path.isdir(base_dir):
        print(f"Error: The base directory {base_dir} for collection '{args.collection_name}' does not exist.")
        sys.exit(1)
    print(f"process from dir {base_dir} to collection {collection_name}")
    while True:
        main_loop(collection_name, base_dir)
        time.sleep(60)
