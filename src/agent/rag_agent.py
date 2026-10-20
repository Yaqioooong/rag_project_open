"""Tool-calling RAG agent with in-memory chat history."""

from __future__ import annotations

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

from llm_models.all_llm import llm
from tools.retriever_tools import retriever_tool

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def build_agent():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个智能助手，尽可能调用工具回答用户的问题。"),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
        ]
    )
    agent = create_tool_calling_agent(llm, [retriever_tool], prompt)
    executor = AgentExecutor(agent=agent, tools=[retriever_tool])
    return RunnableWithMessageHistory(
        executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


def run_agent(query: str, session_id: str = "default"):
    agent_with_history = build_agent()
    return agent_with_history.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}},
    )


def main() -> None:
    response = run_agent("什么是EUV光刻机？")
    print(response)


if __name__ == "__main__":
    main()
