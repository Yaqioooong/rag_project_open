import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import MilvusClient, DataType, Function, FunctionType, IndexType
from pymilvus.client.types import MetricType

from documents.markdown_parser import MarkdownParser
from llm_models.embeddings_model import bge_embedding
from utils.env_utils import COLLECTION_NAME, MILVUS_LITE_URI, PROJECT_ROOT


class MilvusVectorSaver:
    """把新的document数据插入到数据库中"""

    def __init__(self) -> object:
        """自定义collection的索引"""
        self.vector_store_saved: Milvus = None

    def create_collection(self):
        Path(MILVUS_LITE_URI).expanduser().resolve().parent.mkdir(
            parents=True, exist_ok=True
        )
        client = MilvusClient(uri=MILVUS_LITE_URI)
        schema = client.create_schema()
        schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=6000, enable_analyzer=True,
                         analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]})
        schema.add_field(field_name='category', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='source', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='filename', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='filetype', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='title', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='category_depth', datatype=DataType.INT64)
        schema.add_field(field_name='sparse', datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name='dense', datatype=DataType.FLOAT_VECTOR, dim=512)

        bm25_function = Function(
            name="text_bm25_emb",  # Function name
            input_field_names=["text"],  # Name of the VARCHAR field containing raw text data
            output_field_names=["sparse"],
            # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
            function_type=FunctionType.BM25,  # Set to `BM25`
        )

        schema.add_function(bm25_function)

        index_params = client.prepare_index_params()

        index_params.add_index(
            field_name="sparse",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",  # Inverted index type for sparse vectors
            metric_type="BM25",
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                # Algorithm for building and querying the index. Valid values: DAAT_MAXSCORE, DAAT_WAND, TAAT_NAIVE.
                "bm25_k1": 1.2,
                "bm25_b": 0.75
            },
        )
        index_params.add_index(
            field_name="dense",
            index_name="dense_inverted_index",
            index_type=IndexType.HNSW,  # Inverted index type for sparse vectors
            metric_type=MetricType.IP,
            params={"M": 16, "efConstruction": 64}  # M :邻接节点数, efConstruction: 搜索范围
        )

        if COLLECTION_NAME in client.list_collections():
            # 删除collection
            client.drop_collection(collection_name=COLLECTION_NAME)

        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )

    def create_connection(self):
        """创建一个Connection： milvus + langchain"""
        self.vector_store_saved = Milvus(
            embedding_function=bge_embedding,
            collection_name=COLLECTION_NAME,
            builtin_function=BM25BuiltInFunction(),
            vector_field=['dense', 'sparse'],
            consistency_level="Strong",
            auto_id=True,
            connection_args={"uri": MILVUS_LITE_URI},
            search_params=[
                {
                    "metric_type": "IP",
                    "params": {"ef": 64}
                },
                {
                    "metric_type": "BM25",
                    "params": {"drop_ratio_build": 0},
                }
            ]
        )

    def add_documents(self, data: List[Document]):
        """把新的document保存到Milvus中"""

        for doc in data:
            if not doc.metadata.get("title"):
                filename = doc.metadata.get("filename", "")
                doc.metadata["title"] = os.path.splitext(filename)[0]
            if doc.metadata.get("category_depth") is None:
                doc.metadata["category_depth"] = 0
        self.vector_store_saved.add_documents(data)


if __name__ == '__main__':
    file_path = str(PROJECT_ROOT / "data" / "md" / "tech_report_0tfhhamx.md")
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)

    # 写入milvus
    ms = MilvusVectorSaver()
    ms.create_collection()
    ms.create_connection()
    ms.add_documents(docs)

    client = ms.vector_store_saved.client
    # 得到表结构
    desc_collection = client.describe_collection(
        collection_name=COLLECTION_NAME
    )
    print('表结构是: ', desc_collection)

    # 得到当前表的，所有的index
    res = client.list_indexes(
        collection_name=COLLECTION_NAME
    )
    print('表中的所有索引：', res)

    if res:
        for i in res:
            # 得到索引的描述
            desc_index = client.describe_index(
                collection_name=COLLECTION_NAME,
                index_name=i
            )
            print(desc_index)

    result = client.query(
        collection_name=COLLECTION_NAME,
        filter="category == 'Title'",  # 查询 category == 'Title' 的所有数据
        output_fields=['text', 'sparse', 'category', 'filename']  # 指定返回的字段
    )

    print('测试 过滤查询的结果是: ', result)
