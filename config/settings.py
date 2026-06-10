"""
config/settings.py
Centralized configuration using Pydantic BaseSettings.
All values read from environment / .env file.
"""

from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # ── API Keys ────────────────────────────────────────────────────────────────
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    tavily_api_key: str = Field(..., env="TAVILY_API_KEY")
    e2b_api_key: str = Field(..., env="E2B_API_KEY")

    # ── LLM Models available via Groq ───────────────────────────────────────────
    # Primary model for agents
    model_llama_70b: str = "llama-3.3-70b-versatile"
    # Faster, cheaper — for benchmark comparison
    model_llama_8b: str = "llama-3.1-8b-instant"
    # Alternative architecture
    model_mixtral: str = "llama-3.1-8b-instant"
    # Vision-capable model
    model_vision: str = "llama-3.2-11b-vision-preview"

    # Default model used by agents
    default_model: str = "llama-3.1-70b-versatile"

    # ── Agent Config ────────────────────────────────────────────────────────────
    max_agent_iterations: int = Field(default=10, env="MAX_AGENT_ITERATIONS")
    agent_timeout_seconds: int = Field(default=60, env="AGENT_TIMEOUT_SECONDS")

    # ── Benchmark Config ────────────────────────────────────────────────────────
    benchmark_query_count: int = Field(default=50, env="BENCHMARK_QUERY_COUNT")
    benchmark_results_dir: str = "data/results"
    benchmark_queries_dir: str = "data/queries"

    # ── Logging ─────────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", env="LOG_LEVEL"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton — import this everywhere
settings = Settings()


# ── Model registry for easy iteration in benchmarks ────────────────────────────
MODEL_REGISTRY = {
    "llama-70b": settings.model_llama_70b,
    "llama-8b": settings.model_llama_8b,
    "mixtral": settings.model_mixtral,
}

# ── Tool names (must match tool .name attributes) ───────────────────────────────
TOOL_NAMES = [
    "web_search",
    "code_executor",
    "vision_analyzer",
    "calculator",
    "wikipedia_search",
]
