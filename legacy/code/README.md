# 示例代码

> 因为数据并未得到授权，所以数据集并未上传到github，请替换为你想处理的数据

## 目录

- `server` 服务端代码
- `data_import` 数据导入代码

## 前期准备

本服务需要使用Qdrant向量数据库，所以需要先安装Qdrant，为了方便可以使用docker启动：
`docker run -p 6333:6333 \
-v $(pwd)/qdrant_storage:/qdrant/storage \
qdrant/qdrant
`

## 关于Qdrant向量数据库

你可以查看Qdrant的官方文档：https://qdrant.tech/documentation/
