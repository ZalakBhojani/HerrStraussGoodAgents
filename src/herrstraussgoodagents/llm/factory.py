from __future__ import annotations

from .base import LLMClient


def get_client(
    provider: str,
    gcp_project: str = "",
    gcp_location: str = "us-central1",
    openai_api_key: str = "",
    anthropic_api_key: str = "",
) -> LLMClient:
    match provider:
        case "vertexai":
            from .vertexai_client import VertexAIClient

            return VertexAIClient(project=gcp_project, location=gcp_location)
        case "openai":
            from .openai_client import OpenAIClient

            return OpenAIClient(api_key=openai_api_key or None)
        case "anthropic":
            from .anthropic_client import AnthropicClient

            return AnthropicClient(api_key=anthropic_api_key or None)
        case _:
            raise ValueError(f"Unknown LLM provider: {provider!r}")
