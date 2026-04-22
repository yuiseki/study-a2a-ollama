"""SSE ストリーミング経由で Task の状態遷移を観察するクライアント.

`ClientFactory.connect()` で接続し、`send_message` が返すイベントストリーム
（Task 状態の更新 / Artifact の更新 / Message）を時系列で表示する。
Task ライフサイクル（SUBMITTED → WORKING → … → COMPLETED）の生の流れを目で追える。
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from uuid import uuid4

import httpx
from a2a.client import ClientConfig, ClientFactory
from a2a.types import (
    Message,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TextPart,
)

TARGET_URL = "http://127.0.0.1:8003"
QUERY = "Rust の所有権モデルを要約して"


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _text_from_message(message) -> str:
    if not message:
        return ""
    for p in message.parts or []:
        root = getattr(p, "root", p)
        text = getattr(root, "text", None)
        if text:
            return text
    return ""


def _format_item(item) -> str:
    if isinstance(item, Message):
        return f"Message text={_text_from_message(item)!r}"

    if isinstance(item, tuple):
        task, update = item
        if isinstance(update, TaskStatusUpdateEvent):
            state = update.status.state.value if update.status and update.status.state else "?"
            text = _text_from_message(update.status.message) if update.status else ""
            return f"TaskStatusUpdateEvent state={state} final={update.final} text={text!r}"
        if isinstance(update, TaskArtifactUpdateEvent):
            return f"TaskArtifactUpdateEvent artifactId={update.artifact.artifact_id}"
        if update is None and isinstance(task, Task):
            state = task.status.state.value if task.status and task.status.state else "?"
            return f"Task snapshot id={task.id} state={state}"

    return f"(unknown) {type(item).__name__}"


async def main() -> None:
    async with httpx.AsyncClient(timeout=180.0) as httpx_client:
        config = ClientConfig(httpx_client=httpx_client, streaming=True)
        client = await ClientFactory.connect(agent=TARGET_URL, client_config=config)
        card = await client.get_card()
        print(f"{_now()} [card] {card.name} ({card.url})")

        message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=QUERY))],
            message_id=str(uuid4()),
        )
        print(f"{_now()} [send] {QUERY}")
        async for item in client.send_message(message):
            print(f"{_now()} [evt ] {_format_item(item)}")


if __name__ == "__main__":
    asyncio.run(main())
