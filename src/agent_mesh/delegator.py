"""Agent B: Calculation Delegator（:8006）.

- 受け取った計算問題を LLM (qwen3.5:4b) で「人間への依頼文」に清書
- Human Oracle（:8004）に A2A クライアントとして委譲する
- Human の応答を自分の Task 応答として返す

このエージェントは A2A の「中継者 + 書き直し」の例。上流（Agent A）から見れば
サーバー、下流（Human Oracle）から見ればクライアント。
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import httpx
import uvicorn
from a2a.client import ClientConfig, ClientFactory
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Message,
    Part,
    Role,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)
from a2a.utils import new_agent_text_message
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

HOST = "127.0.0.1"
PORT = 8006
HUMAN_URL = "http://127.0.0.1:8004"

MODEL = "qwen3.5:4b"
SYSTEM_PROMPT = """あなたは計算問題を人間に丁寧に依頼する文に清書するアシスタントです。
次のルールを守って 1 文の短い依頼文だけを出力してください:
- 入力の計算式（例: 3 + 5）を必ずそのまま含める
- 親しみやすい日本語で「〜を教えていただけますか」調
- 余計な前置きや装飾、絵文字は不要
"""


def _build_llm() -> ChatOllama:
    return ChatOllama(
        model=MODEL,
        reasoning=False,
        num_predict=128,
        temperature=0.3,
    )


async def _prettify(llm: ChatOllama, question: str) -> str:
    response = await llm.ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]
    )
    return response.content.strip().splitlines()[0]


def _text_from_message(message) -> str | None:
    if not message:
        return None
    for p in message.parts or []:
        root = getattr(p, "root", p)
        text = getattr(root, "text", None)
        if text:
            return text
    return None


async def _ask_human(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=None) as httpx_client:
        config = ClientConfig(httpx_client=httpx_client, streaming=True)
        client = await ClientFactory.connect(agent=HUMAN_URL, client_config=config)
        msg = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=prompt))],
            message_id=str(uuid4()),
        )
        final: str | None = None
        async for item in client.send_message(msg):
            if isinstance(item, tuple):
                _t, update = item
                if isinstance(update, TaskStatusUpdateEvent):
                    if update.status and update.status.state.value == "completed":
                        final = _text_from_message(update.status.message)
        return final or "(no answer)"


class DelegatorExecutor(AgentExecutor):
    def __init__(self) -> None:
        self.llm = _build_llm()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or str(uuid4())
        context_id = context.context_id or str(uuid4())
        updater = TaskUpdater(event_queue, task_id, context_id)

        await updater.submit()
        await updater.start_work()
        await asyncio.sleep(0.1)

        question = context.get_user_input()

        await updater.update_status(
            TaskState.working,
            message=new_agent_text_message(
                f"[Agent B] qwen3.5:4b で人間への依頼文に清書中",
                context_id=context_id,
                task_id=task_id,
            ),
        )
        pretty_prompt = await _prettify(self.llm, question)

        await updater.update_status(
            TaskState.working,
            message=new_agent_text_message(
                f"[Agent B] 清書結果: {pretty_prompt}",
                context_id=context_id,
                task_id=task_id,
            ),
        )

        await updater.update_status(
            TaskState.working,
            message=new_agent_text_message(
                f"[Agent B] Human Oracle に計算を委託",
                context_id=context_id,
                task_id=task_id,
            ),
        )

        answer = await _ask_human(pretty_prompt)

        await updater.update_status(
            TaskState.working,
            message=new_agent_text_message(
                f"[Agent B] Human から回答を受領: {answer}",
                context_id=context_id,
                task_id=task_id,
            ),
        )

        await updater.complete(
            message=new_agent_text_message(
                answer, context_id=context_id, task_id=task_id
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancellation is not supported")


def build_agent_card() -> AgentCard:
    skill = AgentSkill(
        id="delegate_calculation",
        name="計算の依頼",
        description=(
            "受け取った計算式を LLM で人間への親しみやすい依頼文に清書し、"
            "calculation スキルを持つ別ノードに送る。"
        ),
        tags=["delegate", "calculation", "llm"],
        examples=["3 + 5 = ?", "7 + 2 = ?"],
    )
    return AgentCard(
        name="Calculation Delegator",
        description=(
            "受け取った計算式を qwen3.5:4b で人間向けの依頼文に変換し、"
            "Human Oracle に取り次ぐ中継エージェント。"
        ),
        url=f"http://{HOST}:{PORT}/",
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


def main() -> None:
    request_handler = DefaultRequestHandler(
        agent_executor=DelegatorExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(
        agent_card=build_agent_card(),
        http_handler=request_handler,
    )
    uvicorn.run(app.build(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
