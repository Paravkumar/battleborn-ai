from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ticket_agent.schemas import ExecutionLog
from ticket_agent.tools import TOOL_SPECS


def execution_log_to_sft_example(log: ExecutionLog) -> dict[str, Any]:
    user_message = {
        "role": "user",
        "content": json.dumps(
            {
                "goal": log.goal,
                "ticket": log.ticket_snapshot,
            },
            indent=2,
        ),
    }
    messages: list[dict[str, Any]] = [user_message]
    for step in log.steps:
        if step.outcome == "skipped":
            continue
        if step.tool == "customer_reply":
            customer_message = step.data.get("message", "")
            messages.append({"role": "user", "content": str(customer_message)})
            continue
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": step.tool,
                            "arguments": step.arguments,
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "name": step.tool,
                "content": json.dumps(
                    {
                        "message": step.message,
                        "outcome": step.outcome,
                        "data": step.data,
                    }
                ),
            }
        )
    if log.final_response:
        messages.append({"role": "assistant", "content": log.final_response})
    return {
        "messages": messages,
        "tools": TOOL_SPECS,
    }


def sft_example_to_training_text(example: dict[str, Any]) -> str:
    parts: list[str] = []
    tools = example.get("tools") or []
    messages = example.get("messages") or []

    if tools:
        parts.append("Available tools:\n" + json.dumps(tools, indent=2, ensure_ascii=False))

    for message in messages:
        role = message.get("role", "unknown")
        content = str(message.get("content") or "").strip()

        if role == "user":
            parts.append("User:\n" + content)
            continue

        if role == "assistant":
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                parts.append("Assistant tool calls:\n" + json.dumps(tool_calls, indent=2, ensure_ascii=False))
            if content:
                parts.append("Assistant:\n" + content)
            continue

        if role == "tool":
            tool_name = message.get("name", "tool")
            parts.append(f"Tool {tool_name}:\n{content}")
            continue

        if content:
            parts.append(f"{role.title()}:\n{content}")

    return "\n\n".join(part for part in parts if part).strip()


def load_log(path: Path) -> ExecutionLog:
    return ExecutionLog.model_validate_json(path.read_text(encoding="utf-8"))
