import uuid
from typing import Literal

from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition

from graph_corrective.corrective_state import AgentState, Grade
from graph_corrective.generate_node import generate
from graph_corrective.get_human_message import get_last_human_message
from graph_corrective.rewrite_node import rewrite
from llm_models.all_llm import llm
from utils.log_utils import log
from utils.print_utils import _print_event


def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    判断检索到的文档是否与问题相关。
    参数:
        state (messages): 当前状态
    返回:
        str: 判断结果，文档是否相关
    """
    log.info("---检查document的相关性---")
    #  带结构化输出的LLM
    llm_with_structured = llm.with_structured_output(Grade, method="json_mode", )

    # 提示词模板
    prompt = PromptTemplate(
        template="""你是一个评估检索文档与用户问题相关性的评分器。\n
                这是检索到的文档：\n\n {context} \n\n
                这是用户的问题：{question} \n
                如果文档包含与用户问题相关的关键词或语义含义，则评为相关。\n
                给出二元评分 'yes' 或 'no' 来表示文档是否与问题相关。
                请只输出一个 JSON 对象，不要输出 Markdown 代码块或其他解释。
                格式必须是：
                {{"binary_score": "yes"}}
                或
                {{"binary_score": "no"}}
                """,
        input_variables=["context", "question"],
    )

    # 处理链
    chain = prompt | llm_with_structured

    messages = state['messages']
    last_message = messages[-1]

    question = get_last_human_message(messages).content
    docs = last_message.content

    scored_result = chain.invoke({'question': question, 'context': docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---输出：文档相关---")
        return "generate"

    else:
        print("---输出：文档不相关---")
        print(score)
        return "rewrite"


def build_graph():
    """Build the corrective RAG graph without running a chat loop on import."""
    from langgraph.prebuilt import ToolNode

    from graph_corrective.agent_node import agent_node
    from tools.retriever_tools import retriever_tool

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )
    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_edge("rewrite", "agent")
    workflow.add_edge("generate", END)
    return workflow.compile(checkpointer=MemorySaver())


def answer(question: str) -> str:
    """Run one question through a fresh corrective RAG session."""
    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    final_state = {}
    printed = set()
    for event in graph.stream(
        {"messages": [("user", question)]},
        config=config,
        stream_mode="values",
    ):
        final_state = event
        _print_event(event, printed)
    messages = final_state.get("messages", [])
    return messages[-1].content if messages else ""


def run_chat() -> None:
    """Run the interactive corrective RAG chat loop."""
    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    printed = set()

    while True:
        question = input("用户: ")
        if question.lower() in {"q", "exit", "quit"}:
            log.info("对话结束，拜拜！")
            break
        for event in graph.stream(
            {"messages": [("user", question)]},
            config=config,
            stream_mode="values",
        ):
            _print_event(event, printed)


def main() -> None:
    run_chat()


if __name__ == "__main__":
    main()
