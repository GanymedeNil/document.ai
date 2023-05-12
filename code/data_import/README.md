# 数据导入至向量数据库
# 需要先配置../config.json

## 安装依赖

`pip install -r requirements.txt`

## 设置OPENAI_API_KEY

`export OPENAI_API_KEY=sk-xxxxxx`
`export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`

## 注意确保已经设置../config.json，里面有collection和对应的索引文件夹信息

## 运行
`python import_data.py --collection_name=xxx`


`python import_news.py --collection_name=xxx`