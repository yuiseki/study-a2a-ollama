"""Discovery Hero（:9000）.

受け取ったタスクに対して、既知のエージェント URL 全てから Agent Card を取得し
（Direct Configuration discovery）、LLM にそれぞれの name / description / skills を
見せて最適な相手を 1 つ選ばせ、A2A クライアントとして呼び出す。

A2A × LLM のキモは「**LLM が Agent Card カタログを見て自律的に相手を選ぶ**」こと。
そのパターンの最小実装。
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import httpx
import uvicorn
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
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
    Task,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)
from a2a.utils import new_agent_text_message
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

HOST = "127.0.0.1"
PORT = 9000

# Direct Configuration discovery: 既知の URL リスト
KNOWN_AGENTS = [f"http://127.0.0.1:{9001 + i}" for i in range(10)] + [
    "http://127.0.0.1:9011",
]

MODEL = "gemma3:4b"
ROUTER_SYSTEM = """あなたはタスクを最適なエージェントに委譲するルーターです。
エージェント一覧の中から、タスクに最もマッチするエージェントの **name**（英語フルネーム）を
1 行で返してください。以下のルールを厳守:

- 余計な前置き・説明・引用符・改行は書かない
- name そのままを 1 行だけ出力
- リストに無い name を出力してはいけない
"""


def _text_from_parts(parts) -> str | None:
    for p in parts or []:
        root = getattr(p, "root", p)
        text = getattr(root, "text", None)
        if text:
            return text
    return None


class HeroExecutor(AgentExecutor):
    def __init__(self) -> None:
        self.llm = ChatOllama(
            model=MODEL,
            reasoning=False,
            num_predict=48,
            temperature=0.1,
        )

    async def _discover_all(
        self, httpx_client: httpx.AsyncClient
    ) -> list[tuple[str, AgentCard]]:
        cards: list[tuple[str, AgentCard]] = []
        for url in KNOWN_AGENTS:
            try:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
                card = await resolver.get_agent_card()
                cards.append((url, card))
            except Exception:
                # 起動していない or 応答しないエージェントはスキップ
                pass
        return cards

    async def _choose_agent(
        self, task_text: str, cards: list[tuple[str, AgentCard]]
    ) -> tuple[str, AgentCard] | None:
        catalog_lines = []
        for _url, card in cards:
            skills_desc = "; ".join(s.description for s in card.skills)
            catalog_lines.append(
                f"- name: {card.name}\n"
                f"  description: {card.description}\n"
                f"  skills: {skills_desc}"
            )
        catalog = "\n".join(catalog_lines)
        prompt = f"タスク: {task_text}\n\nエージェント一覧:\n{catalog}"

        response = await self.llm.ainvoke(
            [
                SystemMessage(content=ROUTER_SYSTEM),
                HumanMessage(content=prompt),
            ]
        )
        chosen_name = response.content.strip().splitlines()[0].strip("「」\"' ")

        # 完全一致優先、無ければ部分一致
        for url, card in cards:
            if card.name == chosen_name:
                return url, card
        for url, card in cards:
            if chosen_name in card.name or card.name in chosen_name:
                return url, card
        return None

    async def _invoke(
        self, httpx_client: httpx.AsyncClient, url: str, task_text: str
    ) -> str:
        config = ClientConfig(httpx_client=httpx_client, streaming=True)
        client = await ClientFactory.connect(agent=url, client_config=config)
        msg = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=task_text))],
            message_id=str(uuid4()),
        )
        final: str | None = None
        async for item in client.send_message(msg):
            if isinstance(item, Message):
                t = _text_from_parts(item.parts)
                if t:
                    final = t
            elif isinstance(item, tuple):
                task = item[0]
                update = item[1]
                if isinstance(update, TaskStatusUpdateEvent):
                    if update.status and update.status.state.value == "completed":
                        t = (
                            _text_from_parts(update.status.message.parts)
                            if update.status.message
                            else None
                        )
                        if t:
                            final = t
                if isinstance(task, Task):
                    if task.status and task.status.state.value == "completed":
                        t = (
                            _text_from_parts(task.status.message.parts)
                            if task.status.message
                            else None
                        )
                        if t:
                            final = t
        return final or "(no answer)"

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or str(uuid4())
        context_id = context.context_id or str(uuid4())
        updater = TaskUpdater(event_queue, task_id, context_id)

        await updater.submit()
        await updater.start_work()
        await asyncio.sleep(0.1)

        task_text = context.get_user_input()

        async with httpx.AsyncClient(timeout=120.0) as httpx_client:
            # 1. Discovery
            await updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    "[Hero] 既知の URL 全てから Agent Card を取得中...",
                    context_id=context_id,
                    task_id=task_id,
                ),
            )
            cards = await self._discover_all(httpx_client)
            await updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"[Hero] {len(cards)} 個のエージェントを発見",
                    context_id=context_id,
                    task_id=task_id,
                ),
            )

            # 2. LLM による選択
            await updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    "[Hero] LLM にタスクと Agent Card 一覧を渡して最適な相手を選ばせる...",
                    context_id=context_id,
                    task_id=task_id,
                ),
            )
            chosen = await self._choose_agent(task_text, cards)

            if chosen is None:
                await updater.failed(
                    message=new_agent_text_message(
                        "[Hero] 適切なエージェントを特定できませんでした",
                        context_id=context_id,
                        task_id=task_id,
                    )
                )
                return

            chosen_url, chosen_card = chosen
            await updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"[Hero] 選択: {chosen_card.name} @ {chosen_url}",
                    context_id=context_id,
                    task_id=task_id,
                ),
            )

            # 3. Invoke
            answer = await self._invoke(httpx_client, chosen_url, task_text)

            await updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"[Hero] 回答を受領",
                    context_id=context_id,
                    task_id=task_id,
                ),
            )

            await updater.complete(
                message=new_agent_text_message(
                    f"選択されたエージェント: {chosen_card.name}\n\n回答:\n{answer}",
                    context_id=context_id,
                    task_id=task_id,
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancellation is not supported")


def build_agent_card() -> AgentCard:
    return AgentCard(
        name="Discovery Hero",
        description=(
            "タスクを受けて、既知のエージェント群から LLM で最適な相手を"
            "発見・選択し、A2A で委譲して結果を返すルーター。"
        ),
        url=f"http://{HOST}:{PORT}/",
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            AgentSkill(
                id="discover_and_delegate",
                name="発見 & 委譲",
                description="タスクから適切な相手を Agent Card 一覧の中で発見し、呼び出す。",
                tags=["discovery", "routing", "delegation"],
                examples=["来月の売上を予測せよ"],
            )
        ],
    )


def main() -> None:
    handler = DefaultRequestHandler(
        agent_executor=HeroExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=build_agent_card(), http_handler=handler)
    uvicorn.run(app.build(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
