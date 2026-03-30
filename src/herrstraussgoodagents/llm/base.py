from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, TypedDict


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMClient(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str: ...

    @abstractmethod
    async def count_tokens(
        self,
        messages: list[Message],
        model: str,
    ) -> int: ...
