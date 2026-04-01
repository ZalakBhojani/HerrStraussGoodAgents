from __future__ import annotations

import vertexai
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerativeModel,
    Part,
)

from .base import LLMClient, LLMResponse, Message, compute_cost


class VertexAIClient(LLMClient):
    def __init__(self, project: str, location: str = "us-central1") -> None:
        vertexai.init(project=project, location=location)
        self._project = project
        self._location = location

    def _build_contents(
        self, messages: list[Message]
    ) -> tuple[str | None, list[Content]]:
        system_instruction: str | None = None
        contents: list[Content] = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append(
                    Content(role=role, parts=[Part.from_text(msg["content"])])
                )
        return system_instruction, contents

    async def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        system_instruction, contents = self._build_contents(messages)
        gemini = GenerativeModel(
            model_name=model,
            system_instruction=system_instruction,
        )
        response = await gemini.generate_content_async(
            contents,
            generation_config=GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count or 0
        output_tokens = usage.candidates_token_count or 0
        return LLMResponse(
            text=response.text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            cost_usd=compute_cost(model, input_tokens, output_tokens),
        )

    async def count_tokens(self, messages: list[Message], model: str) -> int:
        system_instruction, contents = self._build_contents(messages)
        gemini = GenerativeModel(
            model_name=model,
            system_instruction=system_instruction,
        )
        if not contents:
            contents = [Content(role="user", parts=[Part.from_text(" ")])]
        response = gemini.count_tokens(contents)
        return response.total_tokens
