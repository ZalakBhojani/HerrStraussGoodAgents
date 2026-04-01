from __future__ import annotations

import tiktoken
from openai import AsyncOpenAI

from .base import LLMClient, LLMResponse, Message, compute_cost


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str | None = None) -> None:
        self._client = AsyncOpenAI(api_key=api_key)

    async def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        return LLMResponse(
            text=response.choices[0].message.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            cost_usd=compute_cost(model, input_tokens, output_tokens),
        )

    async def count_tokens(self, messages: list[Message], model: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        total = 0
        for msg in messages:
            total += 4  # per-message overhead
            total += len(encoding.encode(msg["content"]))
        total += 2  # reply primer
        return total
