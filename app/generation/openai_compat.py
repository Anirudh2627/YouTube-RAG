from __future__ import annotations

import httpx

from app.generation.base import LLM
from app.utils.logging import get_logger

log = get_logger(__name__)


class LLMError(RuntimeError):
    pass


class OpenAICompatLLM(LLM):
    name = "openai-compatible"

    def __init__(self, model: str, api_key: str | None, base_url: str | None,
                 default_temperature: float = 0.1, default_max_tokens: int = 1024,
                 timeout_s: float = 60.0, max_retries: int = 2):
        if not api_key:
            raise LLMError(
                "No LLM API key configured. Set LLM_API_KEY in .env "
                "(or LLM_PROVIDER=mock for offline development)."
            )
        self.model = model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client = httpx.Client(
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    
    def complete(self, messages: list[dict], temperature: float | None = None,
                 max_tokens: int | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.default_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.default_max_tokens,
        }
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post(f"{self.base_url}/chat/completions", json=payload)
                if resp.status_code == 429 and attempt < self.max_retries:
                    import time
                    time.sleep(1.5 * (attempt + 1))   # simple backoff on rate limit
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                last_err = e
                log.warning("LLM call failed (attempt %d): %s", attempt + 1, e)
        raise LLMError(f"LLM request failed after retries: {last_err}")
