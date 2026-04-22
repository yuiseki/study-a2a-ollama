"""ナンセンス Agent のファクトリ.

「何を聞かれても好きな○○を返す」だけの無意味な A2A サーバーを topic 毎に量産できる。
discover_demo.py がこれを使って 10 種類のナンセンスエージェントを立ち上げる。

これらは Discovery Hero の「LLM がスキル説明を見て正しい相手を選べるか」を試すための
**おとり**。単体では動作確認以上の意味は無い。
"""
from __future__ import annotations

from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

MODEL = "gemma3:4b"


def _build_llm() -> ChatOllama:
    return ChatOllama(
        model=MODEL,
        reasoning=False,
        num_predict=64,
        temperature=0.8,
    )


class NonsenseExecutor(AgentExecutor):
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.llm = _build_llm()
        self.system_prompt = (
            f"あなたは何を聞かれても『私の好きな{topic}は○○です』とだけ答える"
            "ナンセンスアシスタントです。余計な説明や挨拶は不要です。"
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or str(uuid4())
        context_id = context.context_id or str(uuid4())
        updater = TaskUpdater(event_queue, task_id, context_id)

        await updater.submit()
        await updater.start_work()

        response = await self.llm.ainvoke(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=context.get_user_input()),
            ]
        )
        await updater.complete(
            message=new_agent_text_message(
                response.content.strip(),
                context_id=context_id,
                task_id=task_id,
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancellation is not supported")


def build_nonsense_card(topic: str, host: str, port: int) -> AgentCard:
    return AgentCard(
        name=f"Favorite {topic} Oracle",
        description=(
            f"何を聞かれても『好きな{topic}』を返すナンセンスエージェント。"
            "実用性ゼロ。売上予測・データ分析・計算などの能力は一切無い。"
        ),
        url=f"http://{host}:{port}/",
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            AgentSkill(
                id=f"favorite_{topic}",
                name=f"好きな{topic}を答える",
                description=(
                    f"何を聞かれても好きな{topic}を 1 つ返すだけ。"
                    "計算・検索・データ取得・予測などは絶対にできない。"
                ),
                tags=["nonsense", topic],
            )
        ],
    )
