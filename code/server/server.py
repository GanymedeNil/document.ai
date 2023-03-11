from flask import Flask
from flask import render_template
from flask import request
from qdrant_client import QdrantClient
import openai
import os

app = Flask(__name__)


def prompt(question, answers):
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
    demo_q = '使用以下段落来回答问题："成人头疼，流鼻涕是感冒还是过敏？"\n1. 普通感冒：您会出现喉咙发痒或喉咙痛，流鼻涕，流清澈的稀鼻涕（液体），有时轻度发热。\n2. 常年过敏：症状包括鼻塞或流鼻涕，鼻、口或喉咙发痒，眼睛流泪、发红、发痒、肿胀，打喷嚏。'
    demo_a = '成人出现头痛和流鼻涕的症状，可能是由于普通感冒或常年过敏引起的。如果病人出现咽喉痛和咳嗽，感冒的可能性比较大；而如果出现口、喉咙发痒、眼睛肿胀等症状，常年过敏的可能性比较大。'
    system = '你是一个医院问诊机器人'
    q = '使用以下段落来回答问题，如果段落内容不相关就返回未查到相关信息："'
    q += question + '"'
    # 带有索引的格式
    for index, answer in enumerate(answers):
        q += str(index + 1) + '. ' + str(answer['title']) + ': ' + str(answer['text']) + '\n'

    """
    system:代表的是你要让GPT生成内容的方向，在这个案例中我要让GPT生成的内容是医院问诊机器人的回答，所以我把system设置为医院问诊机器人
    前面的user和assistant是我自己定义的，代表的是用户和医院问诊机器人的示例对话，主要规范输入和输出格式
    下面的user代表的是实际的提问
    """
    res = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': demo_q},
        {'role': 'assistant', 'content': demo_a},
        {'role': 'user', 'content': q},
    ]
    return res


def query(text):
    """
    执行逻辑：
    首先使用openai的Embedding API将输入的文本转换为向量
    然后使用Qdrant的search API进行搜索，搜索结果中包含了向量和payload
    payload中包含了title和text，title是疾病的标题，text是摘要
    最后使用openai的ChatCompletion API进行对话生成
    """
    client = QdrantClient("127.0.0.1", port=6333)
    collection_name = "data_collection"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    sentence_embeddings = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    """
    因为提示词的长度有限，所以我只取了搜索结果的前三个，如果想要更多的搜索结果，可以把limit设置为更大的值
    """
    search_result = client.search(
        collection_name=collection_name,
        query_vector=sentence_embeddings["data"][0]["embedding"],
        limit=3,
        search_params={"exact": False, "hnsw_ef": 128}
    )
    answers = []
    tags = []

    """
    因为提示词的长度有限，每个匹配的相关摘要我在这里只取了前300个字符，如果想要更多的相关摘要，可以把这里的300改为更大的值
    """
    for result in search_result:
        if len(text) > 300:
            summary = text[:300]
        else:
            summary = text
        answers.append({"title": result.payload["title"], "text": summary})

    completion = openai.ChatCompletion.create(
        temperature=0.7,
        model="gpt-3.5-turbo",
        messages=prompt(text, answers),
    )

    return {
        "answer": completion.choices[0].message.content,
        "tags": tags,
    }


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    search = data['search']

    res = query(search)

    return {
        "code": 200,
        "data": {
            "search": search,
            "answer": res["answer"],
            "tags": res["tags"],
        },
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
