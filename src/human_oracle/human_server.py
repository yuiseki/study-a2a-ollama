"""人間を A2A ノードとして扱う実験的サーバー（:8004）.

TASK_STATE_INPUT_REQUIRED で一時停止する従来モデルではなく、「人間」そのものに
独立した Agent Card を持たせて A2A メッシュに参加させるという突飛な発想の検証。

このサーバーは受信した Task を標準出力に表示し、stdin から人間オペレーターの
応答を読み取って Task を完了させる。LLM は呼ばない。Agent Card から見れば
単なる「とても遅いエージェント」として透過的に振る舞う。

前提: このサーバーを起動しているターミナルで人間が応答することになるため、
stdin / stdout が tty に繋がっている状態で実行すること。
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
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message

HOST = "127.0.0.1"
PORT = 8004


class HumanExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or str(uuid4())
        context_id = context.context_id or str(uuid4())
        updater = TaskUpdater(event_queue, task_id, context_id)

        await updater.submit()
        await updater.start_work()

        # submit / start_work のイベントが SSE 経由で client まで届くのを待つ。
        # 単一プロセスで client と stdout を共有するデモのとき、これが無いと
        # 同期 print() が先に走って prompt がイベントログより前に出てしまう。
        await asyncio.sleep(0.1)

        query = context.get_user_input()

        print("\n" + "=" * 60, flush=True)
        print(f"[human oracle] incoming task_id={task_id}", flush=True)
        print("-" * 60, flush=True)
        print(query, flush=True)
        print("-" * 60, flush=True)
        print("応答を 1 行で入力して Enter. (Ctrl+C で reject)", flush=True)

        # stdin は同期的にブロックするので別スレッドに逃がす
        loop = asyncio.get_running_loop()
        try:
            reply = await loop.run_in_executor(None, input, "answer: ")
        except (EOFError, KeyboardInterrupt):
            await updater.reject(
                message=new_agent_text_message(
                    "human operator refused to answer",
                    context_id=context_id,
                    task_id=task_id,
                )
            )
            return

        reply = reply.strip() or "(no comment)"
        print("=" * 60 + "\n", flush=True)

        await updater.complete(
            message=new_agent_text_message(
                reply, context_id=context_id, task_id=task_id
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancellation is not supported")


def build_agent_card() -> AgentCard:
    skill = AgentSkill(
        id="ask_human",
        name="人間オペレーターへの問い合わせ",
        description=(
            "質問を人間に提示し、その回答を返す。応答時間は数秒から数時間。"
            "AI エージェントが自信を持てない判断・ポリシー決定・レビュー等で使う。"
        ),
        tags=["human-in-the-loop", "oracle", "slow"],
        examples=[
            "この設計レビューを承認してよいですか？",
            "この文面は社外公開して問題ありませんか？",
            "今夜の夕食は何が食べたいですか？",
        ],
    )
    return AgentCard(
        name="Human Oracle",
        description=(
            "人間を 1 ノードとして A2A メッシュに公開する実験的エージェント。"
            "LLM は含まず、stdin から応答を得る。"
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
        agent_executor=HumanExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(
        agent_card=build_agent_card(),
        http_handler=request_handler,
    )
    uvicorn.run(app.build(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
