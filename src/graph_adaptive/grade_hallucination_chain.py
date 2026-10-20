from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm_models.all_llm import llm


class GradeHallucinations(BaseModel):
    """对生成回答中是否存在幻觉进行二元评分"""
    binary_score: str = Field(
        description="回答是否基于事实，取值为'yes'或'no'"
    )


# 初始化带函数调用的LLM
structured_llm_grader = llm.with_structured_output(GradeHallucinations, method="json_mode")

# 提示词模板
system = """你是一个评估生成内容是否基于检索事实的评分器。\n
     给出'yes'或'no'的二元评分。'yes'表示回答是基于/支持于给定事实集的。 \n
     请只输出一个 JSON 对象，不要输出 Markdown 代码块或其他解释。
     格式必须是：
     {{"binary_score": "yes"}}
     或
     {{"binary_score": "no"}}
"""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),  # 系统角色设定
        ("human", "事实集：\n\n {documents} \n\n 生成内容: {generation}")
    ]
)

# 构建幻觉检测工作流
hallucination_grader_chain = (
        hallucination_prompt | structured_llm_grader
)
