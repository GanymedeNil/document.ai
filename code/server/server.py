from flask import Flask
from flask import render_template
from flask import request
from qdrant_client import QdrantClient
from flask import jsonify
import openai
import os
from datetime import datetime
from langchain.text_splitter import TokenTextSplitter
import time
from flask_socketio import SocketIO
from flask_socketio import join_room

from flask import Flask, request, session, render_template, redirect, url_for, send_from_directory
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length
from flask_login import login_required
from werkzeug.utils import safe_join
from qdrant_client.http.models import models 
import fnmatch


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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


from langchain.callbacks.base import CallbackManager
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
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

def split_text_to_chunks(text, max_tokens=1000):
    text_splitter = TokenTextSplitter(chunk_size=max_tokens, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    return texts


"""
示例
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
    system = '你是我的财经投资分析师。'
    q = f"请根据下面的数据库信息，整理分析总结并回答我的问题：\"{question}\"。要求：利用和我问题相关的段落回答我的问题,忽略不相关的段落。段落开始：\n"
    q += f" {context} 段落结束。"
    print(q)

    return (system, q)

def query_single(client_id, question, context):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", callback_manager=CallbackManager([StreamingSocketIOCallbackHandler(socketio, client_id)]), streaming=True, verbose=True, max_tokens=460)
    system_msg, prompt_text = prompt(question, context)
    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=prompt_text)
    ]
    msg = llm(messages, client_id)
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
from qdrant_client.conversions import common_types as types
def query(client_id, text, collection="data_collection", selected_files=None):
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
    score_threshold = 0.5
    # 根据选择的文件名创建过滤器
    if len(selected_files) > 0:
        score_threshold = 0.1 # 既然选择文件了，那就一定要搜索出来东西
        query_filter=models.Filter(
            should=[
                models.FieldCondition(
                    key='file_name',
                    match=models.MatchValue(value=filename)
                    ) for filename in selected_files
            ]
        )
    else:
        query_filter = None
        
    print(selected_files, query_filter, f"score_threshold={score_threshold}")
    search_result = client.search(
        collection_name=collection_name,
        query_vector=sentence_embeddings,
        limit=20,
        score_threshold=score_threshold,
        query_filter=query_filter,
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
        """
        PointStruct(id=point_id, vector=embedding_vector, payload={
                                            "title": chunk_summary, 
                                            "text": chunk_text, 
                                            "file_summary": chunk0_summary,
                                            "file_type":file_type, 
                                            "file_dir":file_dir, 
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
    completion = query_single(client_id, question, completion_texts[0])
    return {
        "answer":completion, 
        "tags": tags,
        "qdrant_results": answers,
    }


@app.route('/')
@localnet_or_login_required
def hello_world():
    return render_template('index.html')

@app.route('/download/<path:file_path>', methods=['GET'])
@localnet_or_login_required
def download(file_path):
    base_path = os.path.join(app.root_path, '../data_import')
    safe_file_path = safe_join(base_path, file_path)
    safe_file_dir, safe_file_name = os.path.split(safe_file_path)
    return send_from_directory(directory=safe_file_dir, path=safe_file_name, as_attachment=False)


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    print(data)
    search = data['search']
    client_id = data['client_id']
    collection = data['collection']
    selected_files = data['selected_files']
    res = query(client_id, search, collection, selected_files)
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
    with open('../config.json', 'r') as f:
        config = json.load(f)
    collection_names = list(config['base_dir'].keys())
    return jsonify({"collections": collection_names})

@app.route('/search_files/<collection_name>/<search_term>')
@localnet_or_login_required
def search_files(collection_name, search_term):
    with open('../config.json', 'r') as f:
        config = json.load(f)

    if collection_name not in config['base_dir']:
        raise KeyError(f"Collection '{collection_name}' is not defined in config file.")
    base_dir = os.path.abspath(os.path.join("../data_import", config['base_dir'][collection_name]))
    
    # 读取file_processing_status.json文件
    with open(os.path.join(base_dir, 'file_processing_status.json'), 'r') as status_file:
        file_processing_status = json.load(status_file)

    matches = []
    for root, dirnames, filenames in os.walk(base_dir):
        for filename in fnmatch.filter(filenames, f'*{search_term}*'):
            file_path = os.path.join(root, filename)
            # 由于配置文件是以./开头的，这里也要，否则匹配不上来
            rel_path = "./" + os.path.relpath(file_path, os.path.join("..", "data_import"))  # 获取相对于data_import的路径
            # print(rel_path)
            # 检查文件是否已完成索引
            if rel_path in file_processing_status and file_processing_status[rel_path]['is_completed']:
                ctime = os.path.getctime(file_path)
                formatted_time = datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')
                matches.append({
                    'file_name': os.path.basename(file_path),
                    'time': formatted_time
                })
    # 按时间倒序排列文件
    matches = sorted(matches, key=lambda x: x['time'], reverse=True)
    matches = matches[:24]  # 只返回前24个文件
    return jsonify({"files": matches})
   
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
