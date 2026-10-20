from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from utils.env_utils import (
    ALIBABA_API_KEY,
    ALIBABA_BASE_URL,
    ALIBABA_EMBEDDING_MODEL,
    BGE_DEVICE,
    BGE_MODEL_NAME,
)

openai_embedding = OpenAIEmbeddings(
    openai_api_key=ALIBABA_API_KEY,
    openai_api_base=ALIBABA_BASE_URL,
    model=ALIBABA_EMBEDDING_MODEL,
)


model_name = BGE_MODEL_NAME
model_kwargs = {"device": BGE_DEVICE}
encode_kwargs = {"normalize_embeddings": True}
bge_embedding = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
