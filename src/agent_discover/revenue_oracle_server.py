"""Revenue Forecast Oracle（:9011）.

Redash から過去の売上データを取得し、来月の売上を予測する **唯一の意味のある**
エージェント。Redash 接続はモックで、LLM に現実的なダミー数値を生成させている。

Discovery Hero はタスク「来月の売上を予測せよ」に対して 10 個のナンセンスの中から
このエージェントを選ぶはず。
"""
from __future__ import annotations

from uuid import uuid4

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

HOST = "127.0.0.1"
PORT = 9011
MODEL = "gemma3:4b"

SYSTEM_PROMPT = """あなたは Redash にアクセスして売上データを取得し、来月の売上を予測するアシスタントです。
実際の Redash 接続はモックなので、ダミーでよいので現実的な数値を 1 回だけ返してください。

フォーマット:
来月の売上予測: ¥XX,XXX,XXX（前月比 ±XX.X%）
根拠: 過去 N ヶ月の売上傾向と季節変動を加味した見込み

- 金額は 500 万〜3000 万円程度の現実的な範囲
- 余計な前置き・注釈・免責事項は書かない
"""


class RevenueOracleExecutor(AgentExecutor):
    def __init__(self) -> None:
        self.llm = ChatOllama(
            model=MODEL,
            reasoning=False,
            num_predict=256,
            temperature=0.5,
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or str(uuid4())
        context_id = context.context_id or str(uuid4())
        updater = TaskUpdater(event_queue, task_id, context_id)

        await updater.submit()
        await updater.start_work()

        response = await self.llm.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
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


def build_agent_card() -> AgentCard:
    return AgentCard(
        name="Revenue Forecast Oracle",
        description=(
            "Redash に接続して過去の売上データを取得し、"
            "来月以降の売上を予測するエージェント（Redash 接続はダミー）。"
        ),
        url=f"http://{HOST}:{PORT}/",
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            AgentSkill(
                id="forecast_next_month_revenue",
                name="来月の売上予測",
                description=(
                    "Redash のダッシュボードから過去の売上データを取得し、"
                    "来月の売上金額を予測して返す。金融 / 経営 / データ分析タスク向け。"
                ),
                tags=["revenue", "forecast", "redash", "analytics", "finance"],
                examples=[
                    "来月の売上を予測せよ",
                    "Q4 の売上見込みは？",
                    "過去 3 ヶ月のトレンドから次月予測を出して",
                ],
            )
        ],
    )


def main() -> None:
    handler = DefaultRequestHandler(
        agent_executor=RevenueOracleExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=build_agent_card(), http_handler=handler)
    uvicorn.run(app.build(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
