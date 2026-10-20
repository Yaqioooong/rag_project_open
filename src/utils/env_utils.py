import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
ALIBABA_API_KEY = os.getenv("ALIBABA_API_KEY")
K2_API_KEY = os.getenv("K2_API_KEY")

K2_BASE_URL = os.getenv("K2_BASE_URL")
ALIBABA_BASE_URL = os.getenv("ALIBABA_BASE_URL")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL")

LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL")
DAYTONA_API_KEY = os.getenv("DAYTONA_API_KEY")
DAYTONA_BASE_URL = os.getenv("DAYTONA_BASE_URL")

DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
ALIBABA_EMBEDDING_MODEL = os.getenv(
    "ALIBABA_EMBEDDING_MODEL", "qwen3.7-text-embedding-flash"
)
BGE_MODEL_NAME = os.getenv("BGE_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
BGE_DEVICE = os.getenv("BGE_DEVICE", "cpu")

MILVUS_URI = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")


def _resolve_lite_uri() -> str:
    configured = os.getenv("MILVUS_LITE_URI")
    if not configured:
        return str(PROJECT_ROOT / "db" / "milvus_demo.db")

    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


MILVUS_LITE_URI = _resolve_lite_uri()
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "t_collection01")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
