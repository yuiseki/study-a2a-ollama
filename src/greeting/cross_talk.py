"""Cross-framework cross-talk demo: 「覚えているか」テスト.

Strands（タメ口）と LangGraph（丁寧語）の両方に対して、同じ人間ユーザーが
以下の 2 往復を試みる:

  1. 「こんにちは、私の名前は西園寺昌之です。あなたのお名前は？」
  2. 「よろしくお願いします。ところで、私の名前を覚えていますか？」

クライアントから見ると「会話の継続」に見えるが、**各 SendMessage は新しい
Task** で contextId も共有していない。A2A プロトコル上は各 Task 独立なので、
理屈の上ではエージェントは前のターンを記憶していないはず。

ところが実験してみると、フレームワーク実装によって**暗黙に記憶が残る**場合が
ある。LangGraph 版（自前 AgentExecutor + checkpointer なし）は本当に忘れるが、
Strands 版（A2AServer(agent=...) wrapper）は Agent インスタンスの会話履歴を
再利用するので**覚えている**ことが観察できる。

これは A2A プロトコルの性質ではなく、フレームワーク実装の副作用。厳密な
stateless 契約を守りたい場合はサーバー側で履歴をクリアする実装を明示的に
書く必要がある。

a2a-sdk の `ClientFactory.connect()` を使った実装。
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import httpx
from a2a.client import ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, Task, TextPart

LANGGRAPH_URL = "http://127.0.0.1:8001"
STRANDS_URL = "http://127.0.0.1:8002"

# わざとかなり具体的な名前にしている。ハルシネーションで偶然当たる確率を下げるため。
INTRO = "こんにちは、私の名前は西園寺昌之（さいおんじ まさゆき）です。あなたのお名前は？"
MEMORY_TEST = (
    "よろしくお願いします。ところで、私の名前を覚えていますか？"
    "漢字のフルネームで答えてください。"
)


def _text_from_parts(parts) -> str | None:
    for p in parts or []:
        root = getattr(p, "root", p)
        text = getattr(root, "text", None)
        if text:
            return text
    return None


def _extract_text(item) -> str | None:
    if isinstance(item, Message):
        return _text_from_parts(item.parts)
    if isinstance(item, tuple):
        task = item[0]
        if isinstance(task, Task):
            if task.artifacts:
                for art in task.artifacts:
                    text = _text_from_parts(art.parts)
                    if text:
                        return text
            if task.status and task.status.message:
                return _text_from_parts(task.status.message.parts)
    return None


async def send(config: ClientConfig, base_url: str, text: str) -> str:
    """毎回新しい Task として送信する（contextId を共有しない）."""
    client = await ClientFactory.connect(agent=base_url, client_config=config)
    message = Message(
        role=Role.user,
        parts=[Part(root=TextPart(text=text))],
        message_id=str(uuid4()),
    )
    reply: str | None = None
    async for item in client.send_message(message):
        t = _extract_text(item)
        if t:
            reply = t
    return reply or "(no text reply)"


async def try_memory(label: str, url: str, config: ClientConfig) -> None:
    print(f"\n=== {label} @ {url} ===")

    # turn 1: 名前を伝えて相手の名前を聞く
    print(f"[user  ] {INTRO}")
    reply1 = await send(config, url, INTRO)
    print(f"[agent ] {reply1}\n")

    # turn 2: 新しい Task で記憶を問う。プロトコル上は覚えていないはず。
    print(f"[user  ] {MEMORY_TEST}")
    reply2 = await send(config, url, MEMORY_TEST)
    print(f"[agent ] {reply2}")


async def main() -> None:
    async with httpx.AsyncClient(timeout=180.0) as httpx_client:
        config = ClientConfig(httpx_client=httpx_client, streaming=True)

        await try_memory("Strands (タメ口)", STRANDS_URL, config)
        await try_memory("LangGraph (丁寧語)", LANGGRAPH_URL, config)

        print(
            "\n"
            "[note] 観察ポイント:\n"
            "  - LangGraph 側 (自前 AgentExecutor + checkpointer なし) は名前を\n"
            "    思い出せない。A2A 的に期待される挙動。\n"
            "  - Strands 側 (A2AServer(agent=...) wrapper) は覚えている可能性が高い。\n"
            "    Strands の Agent インスタンスがプロセス内で会話履歴を保持し、\n"
            "    A2AServer がそれを再利用するため。\n"
            "  - つまり A2A プロトコル自体は Task ごとステートレスでも、\n"
            "    フレームワーク実装が暗黙に stateful にしてしまうことがある。\n"
            "    stateless を厳密に守りたければサーバー実装で明示的に履歴を破棄する。"
        )


if __name__ == "__main__":
    asyncio.run(main())
