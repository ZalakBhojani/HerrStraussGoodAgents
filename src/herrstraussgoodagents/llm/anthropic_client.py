from __future__ import annotations

import tiktoken
from anthropic import AsyncAnthropic

from .base import LLMClient, Message


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str | None = None) -> None:
        self._client = AsyncAnthropic(api_key=api_key)

    def _split_messages(
        self, messages: list[Message]
    ) -> tuple[str, list[dict[str, str]]]:
        system = ""
        turns: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                turns.append({"role": msg["role"], "content": msg["content"]})
        return system, turns

    async def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        system, turns = self._split_messages(messages)
        kwargs: dict = dict(
            model=model,
            messages=turns,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if system:
            kwargs["system"] = system
        response = await self._client.messages.create(**kwargs)
        return response.content[0].text

    async def count_tokens(self, messages: list[Message], model: str) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        total = 0
        for msg in messages:
            total += 4
            total += len(encoding.encode(msg["content"]))
        total += 2
        return total
