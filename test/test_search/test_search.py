from documents.milvus_db import MilvusVectorSaver

if __name__ == '__main__':
    mv = MilvusVectorSaver()
    mv.create_connection()
    result = mv.vector_store_saved.similarity_search_with_score(
        query='现在最先进的纳米清洗技术是什么',
        k=2
    )
