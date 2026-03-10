"""
Centralized configuration. Loads from .env and validates.

Manages LLM provider selection, API keys, LangSmith tracing, and
data directory settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

import logging

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    All configuration lives here. Loaded from .env file automatically.
    """

    # --- LLM ---
    llm_provider: str = Field(default="", description="'anthropic' or 'openai'. Auto-detected from API keys if empty.")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    llm_model: str = Field(default="", description="Model name. Defaults to gpt-4o (OpenAI) or claude-sonnet-4-20250514 (Anthropic).")

    # --- LangSmith ---
    # Supports both LANGCHAIN_API_KEY and LANGSMITH_API_KEY (common in older setups)
    langchain_tracing_v2: bool = Field(default=True)
    langchain_api_key: str = Field(default="")
    langsmith_api_key: str = Field(default="")
    langchain_project: str = Field(default="spatialchat")
    langsmith_project: str = Field(default="")
    langchain_endpoint: str = Field(default="https://api.smith.langchain.com")

    @property
    def resolved_langsmith_key(self) -> str:
        """Support both LANGCHAIN_API_KEY and LANGSMITH_API_KEY env vars."""
        return self.langchain_api_key or self.langsmith_api_key

    @property
    def resolved_langsmith_project(self) -> str:
        """Support both LANGCHAIN_PROJECT and LANGSMITH_PROJECT env vars."""
        return self.langchain_project if self.langchain_project != "spatialchat" else (self.langsmith_project or self.langchain_project)

    # --- Data ---
    data_dir: Path = Field(default=Path("./data/cache"))
    max_loaded_datasets: int = Field(default=3)

    # --- Sub-agent model (optional, uses a cheaper model for tool-calling agents) ---
    sub_agent_model: str = Field(default="", description="Model for sub-agents. If empty, uses the main model. Set to e.g. gpt-4o-mini to save tokens.")

    # --- App ---
    streamlit_port: int = Field(default=8501)
    log_level: str = Field(default="INFO")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @field_validator("data_dir")
    @classmethod
    def ensure_data_dir_exists(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def resolved_provider(self) -> str:
        """Resolve the LLM provider from explicit setting or API key presence."""
        if self.llm_provider:
            return self.llm_provider
        if self.openai_api_key:
            return "openai"
        if self.anthropic_api_key:
            return "anthropic"
        raise ValueError(
            "No LLM provider configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file."
        )

    @property
    def resolved_model(self) -> str:
        """Resolve the model name, using provider-specific defaults."""
        if self.llm_model:
            return self.llm_model
        defaults = {"openai": "gpt-4o", "anthropic": "claude-sonnet-4-20250514"}
        return defaults[self.resolved_provider]

    def get_llm(self):
        """
        Factory method: returns the right LangChain chat model based on config.

        Auto-detects provider from whichever API key is set. Explicitly setting
        LLM_PROVIDER and LLM_MODEL in .env overrides the defaults.
        """
        provider = self.resolved_provider
        model = self.resolved_model

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model,
                anthropic_api_key=self.anthropic_api_key,
                temperature=0,
                max_tokens=4096,
            )
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                openai_api_key=self.openai_api_key,
                temperature=0,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. Use 'anthropic' or 'openai'.")

    def get_sub_agent_llm(self):
        """
        Get the LLM for sub-agents (dataset_finder, exploratory, etc.).

        If SUB_AGENT_MODEL is set in .env, uses that model (e.g. gpt-4o-mini
        for cheaper tool-calling). Otherwise falls back to the main model.
        """
        if not self.sub_agent_model:
            return self.get_llm()

        provider = self.resolved_provider
        model = self.sub_agent_model

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                openai_api_key=self.openai_api_key,
                temperature=0,
            )
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model,
                anthropic_api_key=self.anthropic_api_key,
                temperature=0,
                max_tokens=4096,
            )
        else:
            return self.get_llm()

    def setup_langsmith(self):
        """
        Configure LangSmith tracing via environment variables.

        LangSmith reads from env vars automatically. We set them here
        so the tracing activates for all LangChain/LangGraph calls.
        """
        api_key = self.resolved_langsmith_key
        project = self.resolved_langsmith_project
        if api_key:
            os.environ.setdefault("LANGCHAIN_TRACING_V2", str(self.langchain_tracing_v2).lower())
            os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
            os.environ.setdefault("LANGSMITH_API_KEY", api_key)
            os.environ.setdefault("LANGCHAIN_PROJECT", project)
            os.environ.setdefault("LANGCHAIN_ENDPOINT", self.langchain_endpoint)
            logger.info(f"LangSmith tracing enabled for project: {project}")
        else:
            logger.info("LangSmith tracing disabled (no API key provided)")


@lru_cache()
def get_settings() -> Settings:
    """Singleton settings instance. Call this anywhere you need config."""
    return Settings()
