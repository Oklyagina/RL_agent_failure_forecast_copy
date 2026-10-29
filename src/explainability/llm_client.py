"""Small, bounded client for the OpenAI-compatible distillation endpoint."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import requests

from project_config import get_optional_config


class LLMUnavailableError(RuntimeError):
    """Raised when the optional LLM service cannot produce a response."""


class _EmptyAssistantResponse(ValueError):
    """Internal signal for a response with no usable final answer."""


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
    enable_thinking: bool | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "LLMConfig":
        """Build settings, resolving an API key through the existing environment."""
        for name in ("api_url", "model"):
            value = values.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"LLM configuration field {name!r} must be a string")
        api_key = values.get("api_key")
        api_key_env = values.get("api_key_env")
        if api_key is not None and not isinstance(api_key, str):
            raise ValueError("LLM configuration field 'api_key' must be a string")
        if api_key_env is not None and not isinstance(api_key_env, str):
            raise ValueError("LLM configuration field 'api_key_env' must be a string")
        if not api_key and api_key_env:
            api_key = get_optional_config(str(api_key_env))
        missing = [name for name in ("api_url", "model") if not values.get(name)]
        if not api_key:
            missing.append(str(api_key_env or "api_key"))
        if missing:
            raise ValueError("Missing LLM configuration: " + ", ".join(missing))

        verify_ssl = values.get("verify_ssl", True)
        if not isinstance(verify_ssl, bool):
            raise ValueError("LLM configuration field 'verify_ssl' must be a bool")
        enable_thinking = values.get("enable_thinking")
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise ValueError(
                "LLM configuration field 'enable_thinking' must be a bool or null"
            )
        config = cls(
            api_url=str(values["api_url"]),
            api_key=str(api_key),
            model=str(values["model"]),
            max_tokens=int(values.get("max_tokens", 4000)),
            timeout=float(values.get("timeout", 180.0)),
            verify_ssl=verify_ssl,
            retries=int(values.get("retries", 3)),
            retry_delay=float(values.get("retry_delay", 3.0)),
            enable_thinking=enable_thinking,
        )
        if config.max_tokens < 1 or config.timeout <= 0 or config.retries < 1:
            raise ValueError("LLM max_tokens, timeout, and retries must be positive")
        if config.retry_delay < 0:
            raise ValueError("LLM retry_delay cannot be negative")
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "LLMConfig":
        """Load settings from the combined project profile or a flat JSON file."""
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("LLM configuration must be a JSON object")
        return cls.from_mapping(values)


class LLMClient:
    """Send one prompt with bounded timeout/retry behavior and concise progress."""

    def __init__(self, config: LLMConfig, session: Any = requests):
        self.config = config
        self.session = session

    def complete(self, prompt: str, *, temperature: float = 0.7) -> str:
        """Return assistant text or raise a clear optional-service error."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        last_error: Exception | None = None
        use_prompt_fallback = False
        for attempt in range(1, self.config.retries + 1):
            request_prompt = (
                f"{prompt.rstrip()}\n\n/no_think"
                if use_prompt_fallback
                else prompt
            )
            payload = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": request_prompt}],
                "temperature": float(temperature),
                "max_tokens": self.config.max_tokens,
            }
            if self.config.enable_thinking is not None and not use_prompt_fallback:
                payload["chat_template_kwargs"] = {
                    "enable_thinking": self.config.enable_thinking,
                }
            started = time.perf_counter()
            try:
                response = self.session.post(
                    self.config.api_url,
                    headers=headers,
                    json=payload,
                    verify=self.config.verify_ssl,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                choice = response.json()["choices"][0]
                message = choice["message"]
                content = message["content"]
                if not isinstance(content, str) or not content.strip():
                    raise _EmptyAssistantResponse(
                        "LLM response contained no final assistant text "
                        f"(finish_reason={choice.get('finish_reason')!r}, "
                        "reasoning_content_present="
                        f"{bool(message.get('reasoning_content'))})"
                    )
                print(
                    f"[explainability] Response received in "
                    f"{time.perf_counter() - started:.2f}s"
                )
                return content
            except (
                requests.RequestException,
                TimeoutError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as exc:
                last_error = exc
                if attempt < self.config.retries:
                    hard_switch_rejected = (
                        isinstance(exc, requests.HTTPError)
                        and exc.response is not None
                        and exc.response.status_code in {400, 422}
                    )
                    switch_to_prompt = (
                        self.config.enable_thinking is False
                        and not use_prompt_fallback
                        and (
                            isinstance(exc, _EmptyAssistantResponse)
                            or hard_switch_rejected
                        )
                    )
                    if switch_to_prompt:
                        use_prompt_fallback = True
                        print(
                            f"[explainability] Attempt {attempt}/"
                            f"{self.config.retries} did not produce a usable "
                            "non-thinking response; switching to /no_think fallback"
                        )
                        continue
                    print(
                        f"[explainability] Attempt {attempt}/{self.config.retries} "
                        f"failed ({type(exc).__name__}); retrying in "
                        f"{self.config.retry_delay:g}s"
                    )
                    time.sleep(self.config.retry_delay)
        raise LLMUnavailableError(
            f"LLM call failed after {self.config.retries} attempts: {last_error}"
        ) from last_error
