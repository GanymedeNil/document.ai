# import os
# import tempfile
# import subprocess
# import docx
# import tqdm
# # ...

# def doc_to_docx(file_path):
#     temp_dir = tempfile.mkdtemp()
#     output_file_path = os.path.join(temp_dir, "converted.docx")
#     result = subprocess.run(["unoconv", "-f", "docx", "-o", output_file_path, file_path])
    
#     if result.returncode != 0:
#         raise RuntimeError(f"Unoconv failed with return code {result.returncode}")
    
#     return output_file_path

# # ...
# # count = 0
# for root, dirs, files in os.walk("./source_data"):
#     for file in tqdm.tqdm(files):
#         file_path = os.path.join(root, file)
#         file_name = os.path.basename(file_path)
#         file_update_time = os.path.getmtime(file_path)
        
#         if file.lower().endswith('.doc'):
#             file_type = "word"
#             try:
#                 converted_file_path = doc_to_docx(file_path)
#             except RuntimeError as e:
#                 print(f"Error converting {file_path}: {e}")
#                 continue
#             doc = docx.Document(converted_file_path)
#             text = '\n'.join([para.text for para in doc.paragraphs])
#             print(text)

#         else:
#             continue

import PyPDF2
import io

def parse_pdf_with_pypdf2(file_path, max_chunk_size=1000):
    with open(file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        num_pages = len(pdf_reader.pages)
        pages = [pdf_reader.pages[i].extract_text() for i in range(num_pages)]

    full_text = "".join(pages)
    return full_text
# 定义函数使用pypdf2解析PDF文件
# def parse_pdf_with_pypdf2(file):
#     with open(file, 'rb') as f:
#         pdf_reader = pypdf2.PdfFileReader(f)
#         content = ""
#         for i in range(pdf_reader.getNumPages()):
#             page = pdf_reader.getPage(i)
#             content += page.extractText()
#         return content

from pdfminer.high_level import extract_text
import re

def filter_chinese_and_punctuations(text):
    # 定义正则表达式，匹配中文、数字和标准的标点符号、英文字符和空格、回车符、制表符等，以及键盘上数字哪一行的所有符号
    pattern = re.compile(r'[\u4e00-\u9fa5\d ，。！？、；：‘’“”（）《》【】\[\]【】a-zA-Z,.!?;:\'"/\\\{\}\(\)\<\>\+\-\*/=~=^`|&#%@_\n\r\t]+')

    # # 定义正则表达式，匹配中文、数字和标准的标点符号、英文字符和空格、回车符、制表符等
    # pattern = re.compile(r'[\u4e00-\u9fa5\d ，。！？、；：‘’“”（）《》【】\[\]【】a-zA-Z,.!?;:()\n\r\t]+')
    # 使用正则表达式过滤文本
    result = pattern.findall(text)
    filtered_text = ''.join(result)
    # 去除中文和中文之间的换行符
    filtered_text = re.sub(r'([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])', r'\1\2', filtered_text)
    return filtered_text

# 读取PDF文件的内容到变量中
file = 'source_data/report_20230425/2023-04-25_东吴证券_电动车2023年4月策略：碳酸锂价格基本见底+上海车展，看好需求恢复！.pdf'
content_pypdf2 = parse_pdf_with_pypdf2(file)
content_pypdf2 = filter_chinese_and_punctuations(content_pypdf2)
# # 输出解析结果
# print("pypdf2解析结果：")
# print(content_pypdf2)

# import re
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
# model_name = "csebuetnlp/mT5_multilingual_XLSum"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# def get_summary(text, summary_length=200):
    

#     input_ids = tokenizer(
#         [WHITESPACE_HANDLER(text)],
#         return_tensors="pt",
#         padding="max_length",
#         truncation=True,
#         max_length=512
#     )["input_ids"]

#     output_ids = model.generate(
#         input_ids=input_ids,
#         max_length=summary_length, #84
#         no_repeat_ngram_size=2,
#         num_beams=4
#     )[0]

#     summary = tokenizer.decode(
#         output_ids,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False
#     )
#     return summary
# def main():
#     for _ in range(10):
#         get_summary(content_pypdf2)
    
# if __name__ == "__main__":    
#     main()
# summary = get_summary(content_pypdf2)
# import cProfile

# cProfile.run("main()")

# content_pdfminer = extract_text(file)
# print("PDFMiner解析结果：")
# print(summary)


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
            print(f"Processing file: {file_path}")
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

if __name__ == "__main__":
    src_folder = "./"  # 替换为需要遍历的文件夹
    duplicate_folder = "duplicate_delete"
    find_duplicate_files_and_move(src_folder, duplicate_folder)
