from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm_models.all_llm import llm


class GradeDocuments(BaseModel):
    """对检索到的文档进行相关性评分的二元判断"""
    binary_score: str = Field(
        description="相关性评分, 只能是'yes' 或 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments, method="json_mode")

# 提示词模板
system = """你是一个评估检索文档与用户问题相关性的评分器。\n
    如果文档包含与用户问题相关的关键词或语义含义，则评为相关。\n
    不需要非常严格的测试，目的是过滤掉错误的检索结果。\n
    给出'yes'或'no'的二元评分来表示文档是否与问题相关。 \n
    请只输出一个 JSON 对象，不要输出 Markdown 代码块或其他解释。
    格式必须是：
    {{"binary_score": "yes"}}
    或
    {{"binary_score": "no"}}
    """

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system),
        ('human', "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# 构建检索评分器工作流
retrieval_grader_chain = grade_prompt | structured_llm_grader  # 组合提示模板和LLM评分器
