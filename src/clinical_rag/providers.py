"""
Provider factory: the single seam that makes models swappable via YAML.

Given config, returns the right LangChain objects (embeddings + chat models).
No other module should import a concrete provider (ChatGroq, HuggingFaceEmbeddings,
etc.) directly; they all go through here. Swapping providers is a config change.

Providers are imported lazily inside each branch so that optional dependencies
(e.g. OpenAI) are never required unless that provider is actually selected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dotenv import load_dotenv

from clinical_rag.config import Settings, load_settings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models.chat_models import BaseChatModel

# Load .env once at import so GROQ_API_KEY / OPENAI_API_KEY are available.
load_dotenv()


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Return the embeddings object for the configured provider"""
    settings = settings or load_settings()
    cfg = settings.retrieval.embeddings

    if cfg.provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=cfg.model,
            model_kwargs={"device": cfg.device},
            encode_kwargs={"normalize_embeddings": cfg.normalize},
        )

    if cfg.provider == "openai":  # optional premium
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=cfg.model)

    raise ValueError(f"Unknown embeddings provider: {cfg.provider!r}")


def get_chat_model(
    provider: str,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> BaseChatModel:
    """Return a chat model for an explicit provider/model (used by eval judge too)"""
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # Ollama names the completion-length param differently
        return ChatOllama(model=model, temperature=temperature, num_predict=max_tokens)

    if provider == "openai":  # optional premium
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)

    raise ValueError(f"Unknown chat provider: {provider!r}")


def get_generation_model(settings: Settings | None = None) -> BaseChatModel:
    """Convenience: build the generation chat model from generation.yaml"""
    settings = settings or load_settings()
    llm = settings.generation.llm
    return get_chat_model(
        provider=llm.provider,
        model=llm.model,
        temperature=llm.temperature,
        max_tokens=llm.max_tokens,
    )
