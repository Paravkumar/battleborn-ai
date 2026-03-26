from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

StepOutcome = Literal["success", "failed", "escalated", "skipped"]
RunStatus = Literal["completed", "escalated", "failed"]


class ExecutionStepRecord(BaseModel):
    step_id: str
    tool: str
    purpose: str
    attempt: int
    arguments: dict[str, Any] = Field(default_factory=dict)
    outcome: StepOutcome
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    retryable: bool = False


class ExecutionLog(BaseModel):
    goal: str
    started_at: datetime
    finished_at: datetime | None = None
    status: RunStatus = "failed"
    final_response: str | None = None
    final_ticket_status: str | None = None
    steps: list[ExecutionStepRecord] = Field(default_factory=list)


class TicketResolutionWorkflow:
    """PS3 Domain 1 workflow with strict tools, retries, replanning, and escalation."""

    def __init__(self) -> None:
        kb_path = Path(os.getenv("KB_PATH", "data\\knowledge_base.json"))
        self.kb_articles = self._load_kb(kb_path)
        self.max_retries_per_step = 2
        self.logs_dir = Path("outputs")

    def _load_kb(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            categories = payload.get("categories", [])
            normalized: list[dict[str, Any]] = []
            for category in categories if isinstance(categories, list) else []:
                if not isinstance(category, dict):
                    continue
                category_name = str(category.get("category_name", ""))
                for item in category.get("items", []) if isinstance(category.get("items", []), list) else []:
                    if not isinstance(item, dict):
                        continue
                    detail_chunks = []
                    if isinstance(item.get("details"), list):
                        detail_chunks.extend(str(x) for x in item["details"])
                    if isinstance(item.get("steps"), list):
                        detail_chunks.extend(str(x) for x in item["steps"])
                    summary = " ".join(
                        part
                        for part in [str(item.get("answer", "")).strip(), *[chunk.strip() for chunk in detail_chunks]]
                        if part
                    )
                    normalized.append(
                        {
                            "article_id": str(item.get("id", "")),
                            "title": str(item.get("question", "")),
                            "summary": summary,
                            "category_name": category_name,
                            "escalation_triggers": item.get("escalation_triggers", []),
                        }
                    )
            return normalized
        return []

    async def ticket_reader(self, message: str) -> dict[str, Any]:
        return {"ticket_id": "SESSION-TICKET", "customer_message": message, "status": "open"}

    async def knowledge_base_query(self, query: str, limit: int = 3) -> dict[str, Any]:
        q = query.lower()
        scored: list[tuple[int, dict[str, Any]]] = []
        for article in self.kb_articles:
            title = str(article.get("title", "")).lower()
            summary = str(article.get("summary", "")).lower()
            score = sum(1 for token in q.split() if token and (token in title or token in summary))
            if score > 0:
                scored.append((score, article))
        scored.sort(key=lambda item: item[0], reverse=True)
        hits = [item[1] for item in scored[: max(1, limit)]]
        return {
            "hits": [
                {
                    "article_id": str(hit.get("article_id", "")),
                    "title": str(hit.get("title", "")),
                    "summary": str(hit.get("summary", "")),
                }
                for hit in hits
            ]
        }

    async def response_composer(self, customer_message: str, kb_hits: list[dict[str, Any]]) -> dict[str, Any]:
        if not kb_hits:
            return {"success": False, "retryable": True, "message": "No KB hits for grounding."}

        cloud_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not cloud_key:
            return {"success": False, "retryable": False, "message": "OPENAI_API_KEY missing."}

        cloud_model = os.getenv("CLOUD_MODEL", "gpt-4.1-mini")
        context = " | ".join(str(item.get("summary", "")) for item in kb_hits[:3])
        prompt = (
            "You are a customer support agent. Use only the provided knowledge context.\n"
            "Give concise, practical steps.\n"
            f"Customer message: {customer_message}\n"
            f"Knowledge context: {context}"
        )
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {cloud_key}"},
                json={
                    "model": cloud_model,
                    "messages": [
                        {"role": "system", "content": "You are a precise support assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                },
            )
            response.raise_for_status()
            payload = response.json()
            body = str(payload["choices"][0]["message"]["content"]).strip()
        return {
            "success": True,
            "body": body,
            "citations": [str(item.get("article_id", "")) for item in kb_hits[:3]],
        }

    async def ticket_updater(self, status: str, resolution_summary: str) -> dict[str, Any]:
        return {"status": status, "resolution_summary": resolution_summary}

    async def escalation_trigger(self, reason: str) -> dict[str, Any]:
        return {
            "status": "escalated",
            "customer_message": "I have escalated your case to a specialist who will follow up shortly.",
            "reason": reason,
        }

    def _plan(self) -> list[dict[str, str]]:
        return [
            {"tool": "ticket_reader", "purpose": "Load current ticket context."},
            {"tool": "knowledge_base_query", "purpose": "Retrieve grounded support knowledge."},
            {"tool": "response_composer", "purpose": "Compose a customer-facing response."},
            {"tool": "ticket_updater", "purpose": "Update ticket status after response."},
        ]

    def _replan_args(self, tool_name: str, attempt: int, state: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "ticket_reader":
            return {"message": state["input_message"]}
        if tool_name == "knowledge_base_query":
            if attempt == 1:
                return {"query": state["ticket"]["customer_message"], "limit": 3}
            return {"query": state["ticket"]["customer_message"].replace("?", " ").strip(), "limit": 5}
        if tool_name == "response_composer":
            return {
                "customer_message": state["ticket"]["customer_message"],
                "kb_hits": state.get("kb_hits", []),
            }
        if tool_name == "ticket_updater":
            return {
                "status": "resolved" if state.get("response_body") else "pending_customer",
                "resolution_summary": (state.get("response_body") or "Awaiting customer confirmation")[:220],
            }
        return {}

    async def run_ticket_workflow(self, message: str) -> dict[str, Any]:
        state: dict[str, Any] = {"input_message": message}
        log = ExecutionLog(goal="Resolve customer support ticket", started_at=datetime.now(timezone.utc))
        plan = self._plan()

        for index, step in enumerate(plan, start=1):
            tool_name = step["tool"]
            purpose = step["purpose"]
            succeeded = False
            last_error = "Unknown failure."
            for attempt in range(1, self.max_retries_per_step + 2):
                args = self._replan_args(tool_name, attempt, state)
                try:
                    result = await getattr(self, tool_name)(**args)
                    if tool_name == "response_composer" and not result.get("success", False):
                        raise ValueError(str(result.get("message", "Composer failed.")))
                    if tool_name == "ticket_reader":
                        state["ticket"] = result
                    elif tool_name == "knowledge_base_query":
                        state["kb_hits"] = result.get("hits", [])
                    elif tool_name == "response_composer":
                        state["response_body"] = result.get("body", "")
                        state["citations"] = result.get("citations", [])
                    elif tool_name == "ticket_updater":
                        state["ticket_status"] = result.get("status", "open")

                    log.steps.append(
                        ExecutionStepRecord(
                            step_id=f"step_{index}",
                            tool=tool_name,
                            purpose=purpose,
                            attempt=attempt,
                            arguments=args,
                            outcome="success",
                            message="Step completed.",
                            data=result if isinstance(result, dict) else {"result": str(result)},
                            retryable=False,
                        )
                    )
                    succeeded = True
                    break
                except Exception as exc:
                    last_error = str(exc)
                    retryable = attempt <= self.max_retries_per_step
                    log.steps.append(
                        ExecutionStepRecord(
                            step_id=f"step_{index}",
                            tool=tool_name,
                            purpose=purpose,
                            attempt=attempt,
                            arguments=args,
                            outcome="failed",
                            message=last_error,
                            data={},
                            retryable=retryable,
                        )
                    )
                    if not retryable:
                        break

            if not succeeded:
                escalation = await self.escalation_trigger(f"{tool_name} failed after retries: {last_error}")
                log.steps.append(
                    ExecutionStepRecord(
                        step_id=f"step_{index}_escalation",
                        tool="escalation_trigger",
                        purpose="Escalate unresolved workflow failure.",
                        attempt=1,
                        arguments={"reason": escalation["reason"]},
                        outcome="escalated",
                        message="Escalated to human.",
                        data=escalation,
                        retryable=False,
                    )
                )
                log.status = "escalated"
                log.final_response = escalation["customer_message"]
                log.final_ticket_status = escalation["status"]
                log.finished_at = datetime.now(timezone.utc)
                return await self._finalize_log(log)

        log.status = "completed"
        log.final_response = state.get("response_body", "")
        log.final_ticket_status = state.get("ticket_status", "resolved")
        log.finished_at = datetime.now(timezone.utc)
        return await self._finalize_log(log)

    async def _finalize_log(self, log: ExecutionLog) -> dict[str, Any]:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        filename = datetime.now(timezone.utc).strftime("workflow_%Y%m%dT%H%M%S%fZ.json")
        log_path = self.logs_dir / filename
        log_path.write_text(json.dumps(log.model_dump(mode="json"), indent=2), encoding="utf-8")
        payload = log.model_dump(mode="json")
        payload["log_path"] = str(log_path)
        return payload

    def get_tools(self) -> dict[str, Any]:
        return {
            "run_ticket_workflow": self,
            "ticket_reader": self,
            "knowledge_base_query": self,
            "response_composer": self,
            "ticket_updater": self,
            "escalation_trigger": self,
        }

