# 开发、CI 与精简版共用的 OpenSearch 镜像：在官方 3.9.0 镜像上安装中文分词插件（开发文档 3.3）。
#   analysis-smartcn：官方插件，中文智能分词（隐马尔可夫模型 + 词典），映射默认使用它
#   analysis-icu：官方插件，ICU 词典分词，作为对比候选
# 插件版本随 OpenSearch 版本发布，与镜像版本严格一致。
FROM opensearchproject/opensearch:3.9.0
RUN /usr/share/opensearch/bin/opensearch-plugin install --batch analysis-smartcn analysis-icu
