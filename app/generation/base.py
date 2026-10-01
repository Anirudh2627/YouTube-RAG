from __future__ import annotations

from abc import ABC, abstractmethod


class Message(dict):
    @staticmethod
    def system(content: str) -> dict:
        return {"role": "system", "content": content}

    @staticmethod
    def user(content: str) -> dict:
        return {"role": "user", "content": content}

    @staticmethod
    def assistant(content: str) -> dict:
        return {"role": "assistant", "content": content}


class LLM(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    def complete(self, messages: list[dict], temperature: float | None = None,
                 max_tokens: int | None = None) -> str:
        """Chat completion → assistant text."""


def get_llm(provider: str, model: str, api_key: str | None = None,
            base_url: str | None = None, temperature: float = 0.1,
            max_tokens: int = 1024, timeout_s: float = 60.0) -> LLM:
    if provider == "mock":
        from app.generation.mock import MockLLM
        return MockLLM()
    if provider in ("groq", "openai-compatible"):
        from app.generation.openai_compat import OpenAICompatLLM
        return OpenAICompatLLM(
            model=model, api_key=api_key, base_url=base_url,
            default_temperature=temperature, default_max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
    raise ValueError(f"unknown llm provider: {provider!r}")
