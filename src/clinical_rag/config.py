"""
Configuration management for clinical-rag-eval-engine.

Loads YAML config files and validates them via Pydantic models.
All configuration is centralized here: no magic numbers elsewhere in the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# -----------------------
# Project root detection
# -----------------------

# This walks upward from this file to find the directory containing pyproject.toml.
# This works whether the code runs installed, from src/, or from the repo root.


def _find_project_root() -> Path:
    """Locate the project root by searching upward for pyproject.toml."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: assume CWD is the project root
    return Path.cwd()


PROJECT_ROOT = _find_project_root()
CONFIG_DIR = PROJECT_ROOT / "config"


# -------------------
# YAML loader helper
# -------------------


def _load_yaml(filename: str) -> dict[str, Any]:
    """Load a YAML file from the config directory."""
    filepath = CONFIG_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    with open(filepath, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ------------------------------------------
# Pydantic config models: one per YAML file
# ------------------------------------------

# --- retrieval.yaml ---


class ChunkingConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] = Field(default_factory=lambda: ["\n\n", "\n", ". ", " "])


class DenseRetrievalConfig(BaseModel):
    embedding_model: str = "text-embedding-3-small"
    collection_name: str = "clinical_guidelines"
    top_k: int = 10


class SparseRetrievalConfig(BaseModel):
    top_k: int = 10


class FusionConfig(BaseModel):
    strategy: str = "rrf"
    rrf_k: int = 60
    final_top_k: int = 5


class RetrievalConfig(BaseModel):
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    dense: DenseRetrievalConfig = Field(default_factory=DenseRetrievalConfig)
    sparse: SparseRetrievalConfig = Field(default_factory=SparseRetrievalConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)


# --- generation.yaml ---


class LLMConfig(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 1024


class PromptConfig(BaseModel):
    active_version: str = "v1_baseline"
    directory: str = "prompts"


class GenerationConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)


# --- eval.yaml ---


class RubricDimension(BaseModel):
    name: str
    description: str
    max_score: int = 5


class DatasetConfig(BaseModel):
    path: str = "data/eval/eval_dataset.json"


class RagasConfig(BaseModel):
    metrics: list[str] = Field(
        default_factory=lambda: [
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
        ]
    )


class ClinicalJudgeConfig(BaseModel):
    model: str = "gpt-4o"
    rubric_dimensions: list[RubricDimension] = Field(default_factory=list)


class EvalResultsConfig(BaseModel):
    output_dir: str = "data/eval/results"


class EvalConfig(BaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    ragas: RagasConfig = Field(default_factory=RagasConfig)
    clinical_judge: ClinicalJudgeConfig = Field(default_factory=ClinicalJudgeConfig)
    results: EvalResultsConfig = Field(default_factory=EvalResultsConfig)


# --- service.yaml ---


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class LangfuseConfig(BaseModel):
    enabled: bool = True
    host: str = "http://localhost:3000"


class MLflowConfig(BaseModel):
    enabled: bool = True
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "clinical-rag-eval"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"


class OllamaConfig(BaseModel):
    enabled: bool = False
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1"


class ServiceConfig(BaseModel):
    api: APIConfig = Field(default_factory=APIConfig)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)


# --------------------------
# Top-level settings object
# --------------------------


class Settings(BaseModel):
    """Aggregated settings loaded from all config YAML files."""

    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)


def load_settings() -> Settings:
    """
    Load and validate all configuration from YAML files.
    Returns a fully-typed Settings object. Missing files fall back to defaults.
    """
    raw: dict[str, Any] = {}

    for key, filename in [
        ("retrieval", "retrieval.yaml"),
        ("generation", "generation.yaml"),
        ("eval", "eval.yaml"),
        ("service", "service.yaml"),
    ]:
        try:
            raw[key] = _load_yaml(filename)
        except FileNotFoundError:
            raw[key] = {}  # fall back to Pydantic defaults

    return Settings(
        retrieval=RetrievalConfig(**raw["retrieval"]),
        generation=GenerationConfig(**raw["generation"]),
        eval=EvalConfig(**raw["eval"]),
        service=ServiceConfig(**raw["service"]),
    )
