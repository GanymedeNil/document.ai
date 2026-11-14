# 应用镜像（开发文档 13.1）：接口进程与后台任务进程是同一个镜像的两个命令，精简版与集群版共用。
#   docker build -t eaer-app:dev .                          核心依赖（Word、HTML 原生解析）
#   docker build --build-arg WITH_DOCLING=1 -t eaer-app:dev .   另装 Docling（PDF、扫描件、PPTX、XLSX、图片；带 PyTorch，镜像大）
#   docker build --build-arg WITH_OTLP=1 -t eaer-app:dev .      另装 OTLP 导出（EAER_TRACE_EXPORTER=otlp）
# 基础镜像按标签加摘要锁定（Python 3.14.10，开发文档 3.3）。
FROM python:3.14.10-slim@sha256:9aaa474097d094040649f2162a627c0049b408f52138d980916788d80f04802f

ARG WITH_DOCLING=0
ARG WITH_OTLP=0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY requirements.txt requirements-otlp.txt requirements-docling.txt ./
RUN pip install -r requirements.txt \
 && if [ "$WITH_OTLP" = "1" ]; then pip install -r requirements-otlp.txt; fi \
 && if [ "$WITH_DOCLING" = "1" ]; then pip install -r requirements-docling.txt; fi

COPY eaer ./eaer
RUN useradd --uid 10001 --create-home eaer && mkdir -p /data/files && chown -R eaer /data
USER eaer
ENV EAER_STORAGE_DIR=/data/files
EXPOSE 8000
# 默认是接口进程；后台任务进程用 python -m eaer.ingestion
CMD ["uvicorn", "eaer.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
