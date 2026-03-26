from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel


class RetrieveResponse(BaseModel):
    article_ids: list[str]
    snippets: list[str]


class SupportToolset:
    """Customer support toolset with lightweight JSON RAG."""

    def __init__(self) -> None:
        kb_path = Path(os.getenv("KB_PATH", "data\\knowledge_base.json"))
        self.kb: list[dict[str, Any]] = []
        if kb_path.exists():
            payload = json.loads(kb_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                self.kb = payload

    async def retrieve_knowledge(self, query: str, limit: int = 3) -> dict[str, Any]:
        """Retrieve top KB snippets for a support query."""
        q = query.lower()
        scored: list[tuple[int, dict[str, Any]]] = []
        for article in self.kb:
            title = str(article.get("title", "")).lower()
            summary = str(article.get("summary", "")).lower()
            text = f"{title} {summary}"
            score = sum(1 for token in q.split() if token and token in text)
            if score > 0:
                scored.append((score, article))
        scored.sort(key=lambda item: item[0], reverse=True)
        top = [item[1] for item in scored[: max(limit, 1)]]
        return RetrieveResponse(
            article_ids=[str(a.get("article_id", "")) for a in top],
            snippets=[str(a.get("summary", "")) for a in top],
        ).model_dump()

    async def compose_support_reply(self, customer_message: str, context_snippets: list[str]) -> str:
        """Compose support reply using cloud model with Ollama fallback."""
        cloud_key = os.getenv("OPENAI_API_KEY", "").strip()
        cloud_model = os.getenv("CLOUD_MODEL", "gpt-4.1-mini")
        prompt = (
            "You are a customer support assistant. Give direct, practical guidance.\n"
            "Do not ask unrelated billing/refund questions unless user asks for that.\n"
            f"Customer message: {customer_message}\n"
            f"Knowledge context: {' | '.join(context_snippets)}"
        )

        if cloud_key:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {cloud_key}"},
                    json={
                        "model": cloud_model,
                        "messages": [
                            {"role": "system", "content": "You are a helpful support agent."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.2,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                return str(payload["choices"][0]["message"]["content"]).strip()

        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{ollama_host}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful support agent."},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                },
            )
            response.raise_for_status()
            payload = response.json()
            return str(payload.get("message", {}).get("content", "")).strip()

    async def answer_support_question(self, message: str) -> dict[str, Any]:
        """End-to-end support answer using retrieve + compose."""
        retrieved = await self.retrieve_knowledge(message, limit=3)
        answer = await self.compose_support_reply(message, retrieved["snippets"])
        return {
            "answer": answer,
            "citations": retrieved["article_ids"],
        }

    def get_tools(self) -> dict[str, Any]:
        return {
            "retrieve_knowledge": self,
            "compose_support_reply": self,
            "answer_support_question": self,
        }

