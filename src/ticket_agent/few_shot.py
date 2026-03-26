from __future__ import annotations

import json
from pathlib import Path


def _load_json_message_content(message: dict[str, object]) -> dict[str, object]:
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


def _load_json_tool_content(message: dict[str, object]) -> dict[str, object]:
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


def load_response_examples(path: Path, limit: int = 2) -> list[dict[str, object]]:
    if limit <= 0 or not path.exists():
        return []

    examples: list[dict[str, object]] = []
    seen_issue_types: set[str] = set()

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue

            messages = payload.get("messages")
            if not isinstance(messages, list):
                continue

            ticket_subject = ""
            issue_type = ""
            draft_body = ""
            citations: list[str] = []
            needs_escalation = False

            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = message.get("role")
                if role == "user":
                    user_payload = _load_json_message_content(message)
                    ticket_payload = user_payload.get("ticket")
                    if isinstance(ticket_payload, dict):
                        subject = ticket_payload.get("subject")
                        if isinstance(subject, str):
                            ticket_subject = subject
                elif role == "tool" and message.get("name") == "classify_ticket":
                    tool_payload = _load_json_tool_content(message)
                    data = tool_payload.get("data")
                    if isinstance(data, dict):
                        raw_issue_type = data.get("issue_type")
                        if isinstance(raw_issue_type, str):
                            issue_type = raw_issue_type
                elif role == "tool" and message.get("name") == "draft_response":
                    tool_payload = _load_json_tool_content(message)
                    data = tool_payload.get("data")
                    if isinstance(data, dict):
                        body = data.get("body")
                        if isinstance(body, str):
                            draft_body = body
                        raw_citations = data.get("citations")
                        if isinstance(raw_citations, list):
                            citations = [item for item in raw_citations if isinstance(item, str)]
                        needs_escalation = bool(data.get("needs_escalation"))

            if not ticket_subject or not draft_body or not citations or needs_escalation:
                continue
            if issue_type in seen_issue_types:
                continue

            examples.append(
                {
                    "ticket_subject": ticket_subject,
                    "issue_type": issue_type or "unknown",
                    "citations": citations,
                    "ideal_reply": draft_body,
                }
            )
            seen_issue_types.add(issue_type)

            if len(examples) >= limit:
                break

    return examples
