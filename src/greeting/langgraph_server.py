"""LangGraph + Ollama を A2A サーバーとして公開する（:8001）.

LangGraph の ReAct エージェントを Ollama の gemma3:1b でホストし、
a2a-sdk の AgentExecutor 経由で HTTP/JSON-RPC エンドポイントを提供する。
"""
from __future__ import annotations

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from a2a.utils import new_agent_text_message
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

HOST = "127.0.0.1"
PORT = 8001
MODEL = "gemma3:4b"
SYSTEM_PROMPT = (
    "あなたは非常に丁寧で格式ばった日本語で応答するアシスタントです。"
    "常に敬語・丁寧語を用い、語尾は「〜ございます」「〜いたします」を多用してください。"
    "返答は 1〜2 文の短い挨拶に留めてください。"
)


def build_graph():
    llm = ChatOllama(
        model=MODEL,
        temperature=0.3,
        num_predict=256,  # 挨拶だけなので短く打ち切る
        reasoning=False,  # qwen3 系向け。gemma3 には無影響だが予防線として
    )
    # ツール無しの最小構成。create_react_agent は tools=[] でも動く。
    return create_react_agent(llm, tools=[])


class LangGraphExecutor(AgentExecutor):
    def __init__(self) -> None:
        self.graph = build_graph()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        query = context.get_user_input()
        result = await self.graph.ainvoke(
            {
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=query),
                ]
            }
        )
        reply = result["messages"][-1].content
        await event_queue.enqueue_event(new_agent_text_message(reply))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancellation is not supported")


def build_agent_card() -> AgentCard:
    skill = AgentSkill(
        id="greet_formally",
        name="丁寧な挨拶",
        description="ユーザーに対して格式ばった丁寧な日本語で挨拶を返します。",
        tags=["greeting", "japanese", "formal"],
        examples=["こんにちは", "自己紹介してください"],
    )
    return AgentCard(
        name="LangGraph Formal Greeter",
        description="LangGraph + Ollama gemma3:1b で動く丁寧な挨拶エージェント",
        url=f"http://{HOST}:{PORT}/",
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


def main() -> None:
    request_handler = DefaultRequestHandler(
        agent_executor=LangGraphExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(
        agent_card=build_agent_card(),
        http_handler=request_handler,
    )
    uvicorn.run(app.build(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
