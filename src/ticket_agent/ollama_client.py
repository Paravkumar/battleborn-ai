from __future__ import annotations

import json
from typing import Any

try:
    from ollama import Client
except ImportError:  # pragma: no cover - handled at runtime
    Client = None


class OllamaUnavailable(RuntimeError):
    pass


class OllamaGateway:
    def __init__(self, host: str) -> None:
        self.host = host
        self._client = Client(host=host) if Client is not None else None

    def is_ready(self) -> bool:
        return self._client is not None

    def chat_text(self, model: str, messages: list[dict[str, str]], schema: dict[str, Any] | None = None) -> str:
        if self._client is None:
            raise OllamaUnavailable("The ollama Python package is not installed.")
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if schema is not None:
            request["format"] = schema
        response = self._client.chat(**request)
        message = getattr(response, "message", None)
        if message is not None and hasattr(message, "content"):
            return str(message.content)
        if isinstance(response, dict):
            return str(response["message"]["content"])
        return str(response.message.content)

    def chat_json(self, model: str, messages: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
        return json.loads(self.chat_text(model=model, messages=messages, schema=schema))
