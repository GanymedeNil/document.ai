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
    return filtered_text

# 读取PDF文件的内容到变量中
file = 'source_data/report_20230425/2023-04-25_东吴证券_电动车2023年4月策略：碳酸锂价格基本见底+上海车展，看好需求恢复！.pdf'
content_pypdf2 = parse_pdf_with_pypdf2(file)
content_pypdf2 = filter_chinese_and_punctuations(content_pypdf2)
# # 输出解析结果
print("pypdf2解析结果：")
print(content_pypdf2)

# content_pdfminer = extract_text(file)
# print("PDFMiner解析结果：")
# print(content_pdfminer)