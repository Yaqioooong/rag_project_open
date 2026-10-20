from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI

from utils.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

llm = ChatOpenAI(
    temperature=0.5,
    model=DEEPSEEK_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    extra_body={"thinking": {"type": "disabled"}},
)


web_search_tool = TavilySearchResults(max_results=2)
