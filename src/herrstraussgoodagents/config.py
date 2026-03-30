from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# YAML Config Models
# ---------------------------------------------------------------------------

class LLMConfig(BaseModel):
    provider: Literal["vertexai", "openai", "anthropic"] = "vertexai"
    model: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_tokens: int = 1024


class PromptSections(BaseModel):
    persona_header: str = ""
    goal_statement: str = ""
    behavioral_guidelines: str = ""
    compliance_rules: str = ""
    conversation_style: str = ""
    opening_script: str = ""


class AgentConfig(BaseModel):
    llm: LLMConfig
    prompt: PromptSections


class PersonaContext(BaseModel):
    name: str
    loan_amount: float
    months_overdue: int
    reason: str


class PersonaConfig(BaseModel):
    id: str
    context: PersonaContext
    agreement_probability: float
    resolution_keywords: list[str] = Field(default_factory=list)
    hangup_keywords: list[str] = Field(default_factory=list)
    system_prompt: str


class MetricConfig(BaseModel):
    name: str
    weight: float
    description: str
    scale_min: float = 0.0
    scale_max: float = 5.0


class EvaluationRubric(BaseModel):
    version: str
    metrics: list[MetricConfig]
    forbidden_phrases: list[str] = Field(default_factory=list)
    compliance_auto_cap: float = 2.0


# ---------------------------------------------------------------------------
# App Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    # GCP / VertexAI
    google_cloud_project: str = ""
    google_cloud_location: str = "us-central1"
    google_application_credentials: str = ""

    # OpenAI
    openai_api_key: str = ""

    # Anthropic
    anthropic_api_key: str = ""

    # Temporal
    temporal_host: str = "localhost:7233"
    temporal_namespace: str = "default"
    temporal_task_queue: str = "collections-queue"

    # Token budgets
    main_context_tokens: int = 2000
    handoff_context_tokens: int = 500

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()


# ---------------------------------------------------------------------------
# YAML Loaders
# ---------------------------------------------------------------------------

CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"


def load_agent_config(agent_name: str, version: str = "v1") -> AgentConfig:
    path = CONFIGS_DIR / "agents" / f"{agent_name}_{version}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return AgentConfig(**data)


def load_persona_config(persona_id: str) -> PersonaConfig:
    path = CONFIGS_DIR / "personas" / f"{persona_id}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return PersonaConfig(id=persona_id, **data)


def load_evaluation_rubric(version: str = "v1") -> EvaluationRubric:
    path = CONFIGS_DIR / "evaluation" / f"rubric_{version}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return EvaluationRubric(**data)


# ---------------------------------------------------------------------------
# LLM Client Factory
# ---------------------------------------------------------------------------

def get_llm_client(llm_config: LLMConfig):
    """Instantiate the appropriate LLM client from an LLMConfig."""
    from herrstraussgoodagents.llm.factory import get_client

    settings = get_settings()
    return get_client(
        provider=llm_config.provider,
        gcp_project=settings.google_cloud_project,
        gcp_location=settings.google_cloud_location,
        openai_api_key=settings.openai_api_key,
        anthropic_api_key=settings.anthropic_api_key,
    )
