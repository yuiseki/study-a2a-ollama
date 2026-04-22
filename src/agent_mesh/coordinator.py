"""Agent A: Calculation Coordinator（:8005）.

- 呼ばれたら LLM (gemma3:4b) で 1 桁の足し算問題を 1 問生成する
- Agent B（Delegator, :8006）に A2A クライアントとして委託する
- Agent B の応答をそのまま上流に返す
"""
from __future__ import annotations

import asyncio
import re
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
PORT = 8005
DELEGATOR_URL = "http://127.0.0.1:8006"

MODEL = "gemma3:4b"
SYSTEM_PROMPT = """あなたは 1 桁の簡単な足し算問題を 1 問だけ出題するアシスタントです。
出力は必ず次のフォーマットの 1 行のみ:

N + M = ?

- N と M は 1〜9 の整数
- 余計な説明・解答・挨拶は書かない
- 「=」の後は必ず「?」
"""

_QUESTION_RE = re.compile(r"([1-9])\s*\+\s*([1-9])\s*=\s*\?")


def _build_llm() -> ChatOllama:
    return ChatOllama(
        model=MODEL,
        reasoning=False,
        num_predict=32,
        temperature=0.8,
    )


async def _generate_question(llm: ChatOllama) -> str:
    """LLM に出題させて、フォーマットに合う形だけを抜き出す."""
    response = await llm.ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content="出題してください")]
    )
    text = response.content.strip()
    m = _QUESTION_RE.search(text)
    if m:
        return f"{m.group(1)} + {m.group(2)} = ?"
    # フォーマット違反時はそのまま返す
    return text.splitlines()[0] if text else "1 + 1 = ?"


def _text_from_message(message) -> str | None:
    if not message:
        return None
    for p in message.parts or []:
        root = getattr(p, "root", p)
        text = getattr(root, "text", None)
        if text:
            return text
    return None


async def _ask_delegator(question: str) -> str:
    async with httpx.AsyncClient(timeout=None) as httpx_client:
        config = ClientConfig(httpx_client=httpx_client, streaming=True)
        client = await ClientFactory.connect(agent=DELEGATOR_URL, client_config=config)
        msg = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=question))],
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


class CoordinatorExecutor(AgentExecutor):
    def __init__(self) -> None:
        self.llm = _build_llm()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or str(uuid4())
        context_id = context.context_id or str(uuid4())
        updater = TaskUpdater(event_queue, task_id, context_id)

        await updater.submit()
        await updater.start_work()
        await asyncio.sleep(0.1)

        # 出題を LLM に生成させる
        question = await _generate_question(self.llm)

        await updater.update_status(
            TaskState.working,
            message=new_agent_text_message(
                f"[Agent A] gemma3:4b が出題: {question}",
                context_id=context_id,
                task_id=task_id,
            ),
        )

        await updater.update_status(
            TaskState.working,
            message=new_agent_text_message(
                "[Agent A] Agent B に委託",
                context_id=context_id,
                task_id=task_id,
            ),
        )

        answer = await _ask_delegator(question)

        await updater.update_status(
            TaskState.working,
            message=new_agent_text_message(
                f"[Agent A] Agent B から回答を受領: {answer}",
                context_id=context_id,
                task_id=task_id,
            ),
        )

        await updater.complete(
            message=new_agent_text_message(
                f"{question} → {answer}",
                context_id=context_id,
                task_id=task_id,
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancellation is not supported")


def build_agent_card() -> AgentCard:
    skill = AgentSkill(
        id="generate_and_solve",
        name="1桁の足し算を出題して解く",
        description="LLM で 1 桁の足し算問題を生成し、下流エージェントに答えを求める。",
        tags=["math", "coordinator"],
        examples=["計算して"],
    )
    return AgentCard(
        name="Calculation Coordinator",
        description="gemma3:4b で 1 桁の足し算問題を出して、Agent B に答えを求めるコーディネータ。",
        url=f"http://{HOST}:{PORT}/",
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


def main() -> None:
    request_handler = DefaultRequestHandler(
        agent_executor=CoordinatorExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(
        agent_card=build_agent_card(),
        http_handler=request_handler,
    )
    uvicorn.run(app.build(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
