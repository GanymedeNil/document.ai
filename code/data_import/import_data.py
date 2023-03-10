from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
import os
import tqdm
import openai


def to_embeddings(items):
    sentence_embeddings = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=items[1]
    )
    return [items[0], items[1], sentence_embeddings["data"][0]["embedding"]]


if __name__ == '__main__':
    client = QdrantClient("127.0.0.1", port=6333)
    collection_name = "data_collection"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # 创建collection
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    count = 0
    for root, dirs, files in os.walk("./source_data"):
        for file in tqdm.tqdm(files):
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                parts = text.split('#####')
                item = to_embeddings(parts)
                client.upsert(
                    collection_name=collection_name,
                    wait=True,
                    points=[
                        PointStruct(id=count, vector=item[2], payload={"title": item[0], "text": item[1]}),
                    ],
                )
            count += 1
