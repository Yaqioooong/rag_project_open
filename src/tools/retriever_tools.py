"""使用 LangChain 对 Milvus 进行稠密/稀疏混合检索。"""

from langchain_core.tools import create_retriever_tool

from documents.milvus_db import MilvusVectorSaver

mv = MilvusVectorSaver()
mv.create_connection()

retriever = mv.vector_store_saved.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,
        "ranker_type": "rrf",
        "ranker_params": {"k": 100},
        "filter": {"category": "content"},
    },
)

retriever_tool = create_retriever_tool(
    retriever,
    "rag_retriever",
    "搜索并返回关于半导体和芯片的信息，内容涵盖封装、测试、光刻胶等。",
)
