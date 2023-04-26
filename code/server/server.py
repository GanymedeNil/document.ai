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
from flask_socketio import join_room
from flask import session

from flask import Flask, request, session, render_template, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length
from flask_login import login_required


app = Flask(__name__)
app.secret_key = os.urandom(24)
# from textrank4zh import TextRank4Sentence


from functools import wraps
from flask import request, redirect, url_for
from flask_login import current_user

def localnet_or_login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        ip = request.remote_addr
        if ip.startswith("192.168.") or current_user.is_authenticated:
            return f(*args, **kwargs)
        else:
            return redirect(url_for('login', next=request.url))
    return decorated_function

date_format = "%Y-%m-%d %H:%M:%S"

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


from langchain.callbacks.base import AsyncCallbackManager,CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import sys
from typing import Any, Dict, List, Union
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from typing import Any



socketio = SocketIO(app)


class StreamingSocketIOCallbackHandler(BaseCallbackHandler):
    def __init__(self, websocket, client_id):
        self.websocket = websocket
        self.client_id = client_id

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        socketio.emit('new_output', {'content': token}, room=self.client_id)
    
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

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


# llm = OpenAI(temperature=0, n=1, streaming=True, callback_manager=CallbackManager([StreamingSocketIOCallbackHandler(socketio)]), verbose=True, max_tokens=460, model_name="gpt-3.5-turbo")

import re

# Summary model
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


import tiktoken
def split_text_to_chunks(text, max_tokens=1000):
    text_splitter = TokenTextSplitter(chunk_size=max_tokens, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    return texts


"""
请根据下面的数据库信息，整理分析总结并回答我的问题：请介绍一下东方电子
要求：1. 下面信息由几个段落组成，段落以日期开始，不同段落之间可能毫无关联。
2. 利用和我问题相关的段落回答我的问题,忽略不相关的段落。
段落开始：
2023-04-25：欧比特：珠海欧比特宇航科技股份有限公司投资者关系活动记录表（编号：2023-002）.pdf 中国投资者关系活动记录表 1. 问:我想了解一下中国宇航电子公司的发展情况? �长的。   3.科技作为公司发展的源动力，请问公司科研实力、研发投入如何？请具体介绍一下研发人员的情况？ 
  答:您好！ 1、公司是一家专业从事嵌入式 SoC/SIP芯片/模块、
航空电子系统、宇航控制系统、人脸识别与智能图像分析、微型飞行器、人工智能、卫星星座及卫星大数据服务平台研制生产的高科技企业，科技是公司发展的源动力。近年来公司在原有产品技术的基础上，持续加强研发投入，不断提升和改进，加快产品升级换代的能力。同时，以市场为导向，开发新产品新工艺外，力争主营产品在各个应用领域内取得重大技术突破。 2、公司在人才方面具备较强的核心竞争力，具体而言，有如下几方面人才：宇航电子人才。经过二十余年的发展，公司已拥有一支由教授、海归博士以及高级工程师组成的高水平研发人才；人工智能产业人才。公司 AI研究院引进粤港澳优秀专精人才，集聚校企科研力量，积极推动公司人工智能算法及相 

2023-04-25：东方电子：调研活动信息20230424.pdf 东方电子股份有限公司投资者关系活动记录表 5、中电普华电力公司的发展趋势,公司对行业的发展有何认识? �工作中处于主导地位。在国网营销 2.0
的升级中，中电普华是重要的承担者，公司与中电普华是紧密的合作伙伴，
公司将加大市场布局力度，争取更多的市场份额。 
5、请问公司面对 AI+电力的发展趋势，有相关的业务布局吗，公司对行业的发展有何认识？ 
人工智能技术近期备受关注，特别是在围绕国家的双碳能源低碳转型和数字化转型的产业发展中， AI作为使能技术在电网数字化转型中非常重要。
电力能源行业在数字化转型方面走在社会其他行业的前列，无论是整体的网络结构、关键技术突破还是 AI技术的落地实践方面都在迅速发展。 
公司在2018年就已经成立了专门的人工智能团队，子公司海颐软件也有专门的研发力量。研究电力和能源领域的应用场景，其中包括负荷预测、故障检测诊断、智能维护和修复、电力市场辅助决策和可再生能源的发电预测等方向。公司参与了可再生能源的发电预测项目，以提高发电预测的准确性，
为电力交易� 
2023-04-24：苏文电能：苏文电能投资者关系活动记录表.pdf中国证券公司苏文电能科技股份有限公司星期一(10月8日)在深圳召开了投资者关系活动活动记录表。 问:苏文电能科技股份有限公司(Sinopharm Inc.) 是位于上海的一家电力公司, 我们从去年9月开始就一直从事这个行业,现在已经成立了数字能源事业部。 问题6. 维之后,去引入了像云计算、大数据物联网的新一代技术。 维的基础上，去引入了像云计算、大数据物联网的新一代的技术。 我们对于用户的电力设施会通过电能物联网的方式进行实时的采集分析和处理， 特别是从设计和epc出发，我们可
以做到细分数据的颗粒度的采集，相较于其他导入数据 de 公司来说， 我们对于用户的电力设施精细化的管理和远程的运维更加稳定，相应也提高了电力设施运行的效率和稳定性，实现了企业电力设施无人值守，也降低了用户用电的维护的成本，提高了效率。 
今年公司也成立了数字能源事业部， 基于我们现在比较专业的团队、拥有的电力数据， 我们将这个平台进行相应的升级，现在客户对于软件方面的需求也大大的提高， 针对这些变化和需求，单独成立数字能源事业部， 会提升这块业务的的效率和对市投资者关系活动记录表苏文电能科技股份有限公司场有更清晰的目标。 
问题6.充电站业务，不少运营商在做这个行业， 我们为什么会就打算从 2023-04-24：普源精电：普源精电科技股份有限公司投资者关系活动记录表.pdf 普源精电科技股份有限公司投资者关系活动纪录表 AI、半导体行业和通信行业的发展,对电子测量仪器行业的需求会不会大幅增加? �半导体的技术研究到半导体芯片的集成应用， 对于电子测量仪器行业会有正面的影响。 
 
Q：AI行业和半导体行业的发展对电子测量仪器需求的促进？通讯行业的需求是否预期会大幅增加？ 
A：AI、半导体、通信这三个领域，都是现在非常热点的领域，而且

段落结束。
"""
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
    system = '你是我的财经投资分析师。'
    # q = '使用以下段落来回答问题，如果段落内容不相关就返回未查到相关信息："'
    # q = 'Answer my questions only using data from the included context below in markdown format, include any relevant media or code snippets, if the answer is not in the text, say I do not know.'
    q = f"请根据下面的数据库信息，整理分析总结并回答我的问题：\"{question}\"。要求：利用和我问题相关的段落回答我的问题,忽略不相关的段落。段落开始：\n"
    q += f" {context} 段落结束。"
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
    return (system, q)

def query_single(client_id, question, context):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", callback_manager=CallbackManager([StreamingSocketIOCallbackHandler(socketio, client_id)]), streaming=True, verbose=True, max_tokens=460)
    print("llm model name:", llm.model_name)
    system_msg, prompt_text = prompt(question, context)
    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=prompt_text)
    ]
    msg = llm(messages, client_id)
    # print("msg", msg, msg.content)
    return msg.content

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
def query(client_id, text, collection="data_collection"):
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
    sentence_embeddings = to_embeddings(texts[0])
    # print("sentence_embeddings", sentence_embeddings)
    """
    因为提示词的长度有限，所以我只取了搜索结果的前三个，如果想要更多的搜索结果，可以把limit设置为更大的值
    """
    search_result = client.search(
        collection_name=collection_name,
        query_vector=sentence_embeddings,
        limit=20,
        score_threshold=0.5,
        search_params={"exact": False, "hnsw_ef": 256}
    )
    if len(search_result) == 0:
        return {
            "answer":[], 
            "tags": [],
            "qdrant_results": [],
        }
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
        
    # completion = ""
    completion_texts = split_text_to_chunks(completion_text, 3500)
    
    if len(texts) == 1:
        question = texts[0]
    elif len(texts) == 2:
        question = texts[1]
    else:
        question = text
    # print(f"completion_texts,question",completion_texts,question)
    completion = query_single(client_id, question, completion_texts[0])
    # for i in range(min(2, len(completion_texts))):
        # completion += query_single(question, completion_texts[i])
        # time.sleep(1)
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
@localnet_or_login_required
def hello_world():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    search = data['search']
    client_id = data['client_id']
    collection = data['collection']
    res = query(client_id, search, collection)
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
           
@socketio.on('join_room')
def handle_join_room(client_id):
    print("join_room", client_id)
    join_room(client_id)
     
import json  
with open('../config.json', 'r') as f:
    config = json.load(f)

users = config.get('users', [])



login_manager = LoginManager()
login_manager.init_app(app)

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=4, max=80)])
    submit = SubmitField('Login')
    

class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

    def __repr__(self):
        return f'<User {self.username}>'
    
def authenticate(username, password):
    for idx, user in enumerate(users):
        if user['username'] == username and user['password'] == password:
            return User(idx, username)
    return None

@login_manager.user_loader
def load_user(user_id):
    user_id = int(user_id)
    if 0 <= user_id < len(users):
        return User(user_id, users[user_id]['username'])
    return None

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        user = authenticate(username, password)
        if user:
            login_user(user)
            return redirect(url_for('hello_world'))
        else:
            return 'Invalid username or password'
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
