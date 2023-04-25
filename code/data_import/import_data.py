from qdrant_client import QdrantClient
# import qdrant_openapi_client
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
from qdrant_client.models import Filter, FieldCondition
import os
import tqdm
import openai
import PyPDF2
import uuid
import unicodedata
import tiktoken
import docx
import docx2txt

# from tiktoken import Tokenizer
# from tiktoken.tokenizer import TokenizerException

namespace = uuid.NAMESPACE_URL

from textrank4zh import TextRank4Sentence

def get_title(text, max_len=300):
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source='all_filters')
    summary_sentences = tr4s.get_key_sentences(num=1)

    if len(summary_sentences) > 0:
        summary = summary_sentences[0]['sentence']
        if len(summary) <= max_len:
            return summary
        else:
            return summary[:max_len] + "..."
    else:
        return text[:max_len] + "..." if len(text) > max_len else text


def to_embeddings(items):
    sentence_embeddings = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=items[1]
    )
    return [items[0], items[1], sentence_embeddings["data"][0]["embedding"]]

# text-embedding-ada-002的最大输入是8196 token
def num_tokens_from_messages(text, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        # print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(text, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        # print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(text, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    return len(encoding.encode(text))
    # num_tokens = 0
    # for message in messages:
    #     num_tokens += tokens_per_message
    #     for key, value in message.items():
    #         num_tokens += len(encoding.encode(value))
    #         if key == "name":
    #             num_tokens += tokens_per_name
    # num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    # return num_tokens

import re
def split_text_to_chunks(text, max_tokens=8000):
    paragraphs = re.split(r'[\n。.]', text)
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        current_length = num_tokens_from_messages(current_chunk)
        paragraph_length = num_tokens_from_messages(paragraph)

        if current_length + paragraph_length <= max_tokens:
            if current_chunk:
                current_chunk += "." if paragraph_length > 0 else ""
            current_chunk += paragraph
        else:
            chunks.append(current_chunk)
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def get_pdf_chunks(file_path, max_chunk_size=8000):
    with open(file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        num_pages = len(pdf_reader.pages)
        pages = [pdf_reader.pages[i].extract_text() for i in range(num_pages)]

    full_text = "".join(pages)
    chunks = split_text_to_chunks(full_text, max_chunk_size)
    print(chunks)
    return chunks


# def get_pdf_pages(file_path):
#     with open(file_path, "rb") as f:
#         pdf_reader = PyPDF2.PdfReader(f)
#         num_pages = len(pdf_reader.pages)
#         pages = [pdf_reader.pages[i].extract_text() for i in range(num_pages)]
#     return pages

if __name__ == '__main__':
    client = QdrantClient("127.0.0.1", port=6333)
    collection_name = "data_collection"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # 创建collection,如果已经创建了就不要改了
    # client.recreate_collection(
    #         collection_name=collection_name,
    #         vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    #     )
    # Check if the collection exists
    try:
        # Get information about existing collection
        collection_exists = client.get_collection(collection_name)
        print("Collection exists")
    except Exception as e:
        print(f"Collection {collection_name} not exists, create it")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    # count = 0
    for root, dirs, files in os.walk("./source_data"):
        for file in tqdm.tqdm(files):
            file_path = os.path.join(root, file)
            file_update_time = os.path.getmtime(file_path)
            file_uuid = str(uuid.uuid5(namespace, f"{file_path}_0")) # check file chunk 0 if exists

            print(f"processing file_path:{file_path} with UUiD:{file_uuid}")
            existing_points = client.retrieve(
                collection_name=collection_name,
                ids=[file_uuid], # check chunk 0 if exists
                with_payload=True
            )

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
                    # file_type = "word"
                    # text = docx2txt.process(file_path)
                    # chunks = split_text_to_chunks(text)
                else:
                    file_type = "text"
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        chunks = split_text_to_chunks(text)

                for file_chunk, chunk_text in enumerate(chunks):
                    point_id = str(uuid.uuid5(namespace, f"{file_path}_{file_chunk}"))
                    title = get_title(chunk_text)
                    item = to_embeddings([title, chunk_text])
                    print(f"inserting...... title:{title} point_id:{point_id} chunk_text:{chunk_text}")
                    client.upsert(
                        collection_name=collection_name,
                        wait=True,
                        points=[
                            # user file_path as id, for future search
                            PointStruct(id=point_id, vector=item[2], payload={"title": item[0], "text": item[1], "filename": file, "file_type":file_type, "file_update_time": file_update_time, "file_path": file_path,"file_uuid":file_uuid, "file_chunk": file_chunk}),
                        ],
                    )
                    # count += 1
    # for root, dirs, files in os.walk("./source_data"):
    #     for file in tqdm.tqdm(files):
    #         file_path = os.path.join(root, file)
    #         with open(file_path, 'r', encoding='utf-8') as f:
    #             text = f.read()
    #             parts = text.split('#####')
    #             item = to_embeddings(parts)
    #             print(f"parts:{parts} item:{item}")
    #             client.upsert(
    #                 collection_name=collection_name,
    #                 wait=True,
    #                 points=[
    #                     PointStruct(id=count, vector=item[2], payload={"title": item[0], "text": item[1]}),
    #                 ],
    #             )
    #         count += 1
