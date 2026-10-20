# 初始化工作流图
from pprint import pprint

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from graph_adaptive.generate_node import generate
from graph_adaptive.grade_answer_chain import answer_grader_chain
from graph_adaptive.grade_document_node import grade_documents
from graph_adaptive.grade_hallucination_chain import hallucination_grader_chain
from graph_adaptive.graph_state import GraphState
from graph_adaptive.query_route_chain import question_router_chain
from graph_adaptive.retriever_node import retrieve
from graph_adaptive.transform_query_node import transform_query
from graph_adaptive.web_search_node import web_search
from utils.log_utils import log


def evaluate_generation(state):
    """
    评估生成结果是否基于文档并正确回答问题
    Args:
        state (dict): 当前图状态，包含问题、文档和生成结果
    Returns:
        str: 下一节点的名称（useful/not useful/not supported）
    """
    log.info("---评估：生成内容是否基于DOC---")  # 阶段标识
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # 检查生成是否基于文档
    score = hallucination_grader_chain.invoke({'documents': documents, 'generation': generation})
    grade = score.binary_score

    if grade == 'yes':
        log.info("---判定：生成内容基于DOC---")
        # 检查是否准确回答问题
        answer_score = answer_grader_chain.invoke({'question': question, 'generation': generation})
        answer_grade = answer_score.binary_score
        if answer_grade == 'yes':
            log.info("---判定：生成内容准确回答问题---")
            return "useful"  # 返回有用结果
        else:  # 如果没有回答问题
            log.info("---判定：生成内容未能准确回答问题---")
            return "not useful"  # 返回无用结果
    else:
        log.info("---判定：生成内容未基于DOC，将重新尝试---")
        return "not supported"  # 返回不支持结果


def decide_to_generate(state):
    """
    决定是生成回答还是重新优化问题

    Args:
        state (dict): 当前图状态，包含问题和过滤后的文档

    Returns:
        str: 下一节点的名称（transform_query或generate）
    """
    log.info("---ASSESS GRADED DOCUMENTS---")  # 阶段标识

    filtered_documents = state["documents"]
    transform_count = state.get("transform_count", 0)
    if not filtered_documents:
        if transform_count >= 2:
            log.info("---决策：所有文档都与问题无关,并且已经循环了2次，转为web查询问题---")
            return "web_search"
        log.info("---决策：所有文档都与问题无关，将转换查询问题---")
        return "transform_query"
    else:
        log.info("---决策：生成最终回答---")
        return "generate"


def route_question(state):
    """
    路由问题到网络搜索或RAG流程
    Args:
        state (dict): 当前图状态，包含用户问题

    Returns:
        str: 下一节点的名称（web_search或vectorstore）
    """
    log.info("---ROUTE QUESTION---")  # 阶段标识
    question = state["question"]
    source = question_router_chain.invoke({"question": question})  # 调用问题路由

    # 根据路由结果决定下一个节点
    if source.datasource == "web_search":
        log.info("---路由到web搜索---")
        return "web_search"
    elif source.datasource == "vectorstore":
        log.info("---路由到RAG系统---")
        return "vectorstore"
    else:
        return "web_search"


def build_graph():
    """Build the adaptive RAG workflow."""
    workflow = StateGraph(GraphState)

    workflow.add_node("web_search", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_to_generate)
    workflow.add_conditional_edges(
        "generate",
        evaluate_generation,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    return workflow.compile()


def answer(question: str) -> str:
    """Run one question through the adaptive RAG workflow."""
    final_state = {}
    for output in build_graph().stream({"question": question}, stream_mode="values"):
        final_state = output
    return final_state.get("generation", "")


def run_chat() -> None:
    """Run the interactive adaptive RAG chat loop."""
    graph = build_graph()
    while True:
        question = input("用户：")
        if question.lower() in {"q", "exit", "quit"}:
            print("对话结束，拜拜！")
            break

        final_state = {}
        for output in graph.stream({"question": question}, stream_mode="values"):
            final_state = output
        pprint(final_state.get("generation", ""))


def main() -> None:
    run_chat()


if __name__ == "__main__":
    main()
