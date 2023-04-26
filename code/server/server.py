from flask import Flask
from flask import render_template
from flask import request
from qdrant_client import QdrantClient
from flask import jsonify
import openai
import os
from datetime import datetime
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
import time
from flask_socketio import SocketIO, emit
# from transformers import GPT2TokenizerFast
# tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


app = Flask(__name__)

# from textrank4zh import TextRank4Sentence

date_format = "%Y-%m-%d %H:%M:%S"

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


from langchain.callbacks.base import AsyncCallbackManager,CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import sys
from typing import Any, Dict, List, Union
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

# chat = ChatOpenAI(streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0)
# resp = chat([HumanMessage(content="Write me a song about sparkling water.")])

from typing import Any
socketio = SocketIO(app)


class StreamingSocketIOCallbackHandler(BaseCallbackHandler):
    def __init__(self, websocket):
        self.websocket = websocket

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        socketio.emit('new_output', {'content': token})
    
    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        pass

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        pass

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        pass

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        pass

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        pass

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        pass

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        pass
        
llm = OpenAI(temperature=0, n=1, streaming=True, callback_manager=CallbackManager([StreamingSocketIOCallbackHandler(socketio)]), verbose=True)
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
def get_summary(text, summary_length=200):
    WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer(
        [WHITESPACE_HANDLER(text)],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=1024
    )["input_ids"]

    output_ids = model.generate(
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

import tiktoken

def split_text_to_chunks(text, max_tokens=1000):
    text_splitter = TokenTextSplitter(chunk_size=max_tokens, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    return texts

def prompt(question, context):
    """
    生成对话的示例提示语句，格式如下：
    demo_q:
    使用以下段落来回答问题，如果段落内容不相关就返回未查到相关信息："成人头疼，流鼻涕是感冒还是过敏？"
    1. 普通感冒：您会出现喉咙发痒或喉咙痛，流鼻涕，流清澈的稀鼻涕（液体），有时轻度发热。
    2. 常年过敏：症状包括鼻塞或流鼻涕，鼻、口或喉咙发痒，眼睛流泪、发红、发痒、肿胀，打喷嚏。
    demo_a:
    成人出现头痛和流鼻涕的症状，可能是由于普通感冒或常年过敏引起的。如果病人出现咽喉痛和咳嗽，感冒的可能性比较大；而如果出现口、喉咙发痒、眼睛肿胀等症状，常年过敏的可能性比较大。
    system:
    你是一个医院问诊机器人
    """
    # demo_q = '使用以下段落来回答问题："成人头疼，流鼻涕是感冒还是过敏？"\n1. 普通感冒：您会出现喉咙发痒或喉咙痛，流鼻涕，流清澈的稀鼻涕（液体），有时轻度发热。\n2. 常年过敏：症状包括鼻塞或流鼻涕，鼻、口或喉咙发痒，眼睛流泪、发红、发痒、肿胀，打喷嚏。'
    # demo_a = '成人出现头痛和流鼻涕的症状，可能是由于普通感冒或常年过敏引起的。如果病人出现咽喉痛和咳嗽，感冒的可能性比较大；而如果出现口、喉咙发痒、眼睛肿胀等症状，常年过敏的可能性比较大。'
    system = 'You are a helpful assistant'
    # q = '使用以下段落来回答问题，如果段落内容不相关就返回未查到相关信息："'
    q = 'You are a helpful assistant. Answer my questions only using data from the included context below in markdown format, include any relevant media or code snippets, if the answer is not in the text, say I do not know.'
    q += f" 问题是：{question} 段落如下：{context} 请用中文回答。"
    # 带有索引的格式
    # for index, answer in enumerate(answers):
    #     q += str(index + 1) + '. ' + str(answer['title']) + ': ' + str(answer['text']) + '\n'
    # q += " 请用中文回答。"
    print(q)
    """
    system:代表的是你要让GPT生成内容的方向，在这个案例中我要让GPT生成的内容是医院问诊机器人的回答，所以我把system设置为医院问诊机器人
    前面的user和assistant是我自己定义的，代表的是用户和医院问诊机器人的示例对话，主要规范输入和输出格式
    下面的user代表的是实际的提问
    """
    return q
    # res = [
    #     {'role': 'system', 'content': system},
    #     # {'role': 'user', 'content': demo_q},
    #     # {'role': 'assistant', 'content': demo_a},
    #     {'role': 'user', 'content': q},
    # ]
    # print(res)
    # return res

def query_single(question, context):
    # chat = ChatOpenAI(streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0)
    # resp = chat([HumanMessage(content="Write me a song about sparkling water.")])
    return llm(prompt(question, context))
    # openai.api_key = os.getenv("OPENAI_API_KEY")

    # completion = openai.ChatCompletion.create(
    #     temperature=0.7,
    #     model="gpt-3.5-turbo",
    #     messages=prompt(question, context),
    # )

    # return completion.choices[0].message.content
from text2vec import SentenceModel    
from langchain.text_splitter import TokenTextSplitter
embed_model = SentenceModel('shibing624/text2vec-base-chinese')
def to_embeddings(text):
    embeddings = embed_model.encode([text])
    return embeddings[0].tolist()
"""
You are a helpful assistant
Answer my questions only using data from the included context below in markdown format, 
include any relevant media or code snippets, if the answer is not in the text, say I do not know.
"""
def query(text, collection="data_collection"):
    """
    text 如果加上|，比如 东方雨虹 | 总结文档， 那么会根据东方雨虹从数据库进行搜索，然后跟gpt问的问题是总结文档，
    执行逻辑：
    首先使用openai的Embedding API将输入的文本转换为向量
    然后使用Qdrant的search API进行搜索，搜索结果中包含了向量和payload
    payload中包含了title和text，title是疾病的标题，text是摘要
    最后使用openai的ChatCompletion API进行对话生成
    """
    texts = text.split("|")
    client = QdrantClient("127.0.0.1", port=6333)
    collection_name = collection
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # sentence_embeddings = openai.Embedding.create(
    #     model="text-embedding-ada-002",
    #     input=texts[0]
    # )
    sentence_embeddings = to_embeddings(texts[0])
    """
    因为提示词的长度有限，所以我只取了搜索结果的前三个，如果想要更多的搜索结果，可以把limit设置为更大的值
    """
    search_result = client.search(
        collection_name=collection_name,
        query_vector=sentence_embeddings,
        limit=20,
        score_threshold=0.5,
        search_params={"exact": False, "hnsw_ef": 128}
    )
    answers = []
    tags = []
    # completions = []
    completion_text = ""
    """
    因为提示词的长度有限，每个匹配的相关摘要我在这里只取了前300个字符，如果想要更多的相关摘要，可以把这里的300改为更大的值
    """
    
    completion_num = 0
    for result in search_result:
        # summary = get_summary(result.payload["text"], 200)
        """
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
        """
        # answers.append({"title": result.payload["title"], "text": summary,"filename":result.payload["filename"],"time":datetime.fromtimestamp(result.payload["file_update_time"]).strftime(date_format)})
        answers.append({"payload":result.payload, "time":datetime.fromtimestamp(result.payload["file_update_time"]).strftime(date_format)})

        if completion_num <= 4:
            file_name = result.payload["file_name"]
            file_summary = result.payload["file_summary"]
            title = result.payload["title"]
            text_chunk = split_text_to_chunks(result.payload["text"], 800)[0]

            if file_name not in completion_text:
                completion_text += f" {file_name}"
            
            if file_summary not in completion_text:
                completion_text += f" {file_summary}"
            completion_text += f" {title} {text_chunk}"
            completion_num += 1
        
    completion = ""
    completion_texts = split_text_to_chunks(completion_text, 3600)
    print(completion_texts)
    if len(texts) == 1:
        question = texts[0]
    elif len(texts) == 2:
        question = texts[1]
    else:
        question = text
    for i in range(min(2, len(completion_texts))):
    # if len(completion_texts) > 1:
        completion += query_single(question, completion_texts[i])
        time.sleep(1)
    #     completion += query_single(text, completion_texts[1])
    # else:
    #     completion = query_single(text, completion_texts[0])
    # completions.append(completion)



    # completion = openai.ChatCompletion.create(
    #     temperature=0.7,
    #     model="gpt-3.5-turbo",
    #     # model="gpt-4",
    #     messages=prompt(text, answers),
    # )
    # combined_answer = "\n\n".join([completion["answer"] for completion in completions])

    return {
        # "answer": completion.choices[0].message.content,
        "answer":completion, 
        "tags": tags,
        "qdrant_results": answers,
    }


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    search = data['search']
    collection = data['collection']
    res = query(search, collection)
    return {
        "code": 200,
        "data": {
            "search": search,
            "answer": res["answer"],
            "tags": res["tags"],
            "qdrant_results": res["qdrant_results"],
        },
    }

@app.route('/collections')
def get_collections():
    client = QdrantClient("127.0.0.1", port=6333)
    collections_response = client.get_collections()
    collection_names = [collection.name for collection in collections_response.collections]

    return jsonify({"collections": collection_names})
            
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
