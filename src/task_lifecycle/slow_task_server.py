"""Task ライフサイクルの観察用 A2A サーバー（:8003）.

3 ステップの疑似「調査タスク」を実行し、各ステップの間に
`TaskUpdater.update_status` で進捗メッセージを emit する。SSE ストリーミング越しに
`submitted` → `working`（複数回）→ `completed` の遷移がクライアントから観察できる。

LLM は呼ばず、教育目的でタイマーのみを使うダミー実装。
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    TaskState,
)
from a2a.utils import new_agent_text_message

HOST = "127.0.0.1"
PORT = 8003

STEPS: list[tuple[str, str, float]] = [
    ("expand", "クエリを 3 個のサブクエリに展開しています", 1.5),
    ("gather", "関連情報を収集しています", 1.5),
    ("synth", "サマリを合成しています", 1.5),
]


class SlowResearchExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or str(uuid4())
        context_id = context.context_id or str(uuid4())
        updater = TaskUpdater(event_queue, task_id, context_id)

        await updater.submit()
        await updater.start_work()

        query = context.get_user_input()

        for step, description, delay in STEPS:
            await asyncio.sleep(delay)
            await updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"[{step}] {description}",
                    context_id=context_id,
                    task_id=task_id,
                ),
            )

        final = (
            f"『{query}』についての調査は 3 ステップで完了しました。"
            "これは教育目的のダミー応答です。"
        )
        await updater.complete(
            message=new_agent_text_message(final, context_id=context_id, task_id=task_id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancellation is not supported")


def build_agent_card() -> AgentCard:
    skill = AgentSkill(
        id="slow_research",
        name="疑似調査タスク",
        description="クエリに対して 3 ステップの疑似研究を実行し、途中経過を逐次通知する。",
        tags=["demo", "streaming", "task-lifecycle"],
        examples=["東京の気候について調べて", "Rust の所有権モデルを要約して"],
    )
    return AgentCard(
        name="Slow Research Demo",
        description="Task ライフサイクル観察用の疑似調査エージェント",
        url=f"http://{HOST}:{PORT}/",
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


def main() -> None:
    request_handler = DefaultRequestHandler(
        agent_executor=SlowResearchExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(
        agent_card=build_agent_card(),
        http_handler=request_handler,
    )
    uvicorn.run(app.build(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
