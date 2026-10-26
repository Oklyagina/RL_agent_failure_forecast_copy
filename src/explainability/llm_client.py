"""Small client for the OpenAI-compatible endpoint used by rule distillation."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import requests


@dataclass(frozen=True)
class LLMConfig:
    """Connection settings for an OpenAI-compatible chat-completions API."""

    api_url: str
    api_key: str
    model: str
    max_tokens: int = 4000
    timeout: float = 180.0
    verify_ssl: bool = True
    retries: int = 3
    retry_delay: float = 3.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "LLMConfig":
        api_key = values.get("api_key")
        api_key_env = values.get("api_key_env")
        if not api_key and api_key_env:
            api_key = os.environ.get(str(api_key_env))
        missing = [name for name in ("api_url", "model") if not values.get(name)]
        if not api_key:
            missing.append(str(api_key_env or "api_key"))
        if missing:
            raise ValueError("Missing LLM configuration: " + ", ".join(missing))
        return cls(
            api_url=str(values["api_url"]),
            api_key=str(api_key),
            model=str(values["model"]),
            max_tokens=int(values.get("max_tokens", 4000)),
            timeout=float(values.get("timeout", 180.0)),
            verify_ssl=bool(values.get("verify_ssl", True)),
            retries=int(values.get("retries", 3)),
            retry_delay=float(values.get("retry_delay", 3.0)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "LLMConfig":
        """Load settings from the JSON format used by the legacy workflow."""
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(values)


class LLMClient:
    """Send one user prompt and return the assistant text."""

    def __init__(self, config: LLMConfig, session: Any = requests):
        self.config = config
        self.session = session

    def complete(self, prompt: str, *, temperature: float = 0.7) -> str:
        """Call chat completions with the bounded retry policy from the legacy code."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "max_tokens": self.config.max_tokens,
        }
        last_error: Exception | None = None
        for attempt in range(1, self.config.retries + 1):
            try:
                response = self.session.post(
                    self.config.api_url,
                    headers=headers,
                    json=payload,
                    verify=self.config.verify_ssl,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return content if isinstance(content, str) else str(content)
            except Exception as exc:  # requests and response-shape errors share retry handling
                last_error = exc
                if attempt < self.config.retries:
                    time.sleep(self.config.retry_delay)
        raise RuntimeError(
            f"LLM call failed after {self.config.retries} attempts: {last_error}"
        ) from last_error
