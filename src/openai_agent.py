from __future__ import annotations

from .agent_toolset import SupportToolset


def create_agent() -> dict[str, object]:
    """Create PS3 Domain 1 ticket-resolution agent definition."""
    toolset = SupportToolset()
    return {
        "tools": toolset.get_tools(),
        "system_prompt": (
            "You are a workflow orchestration agent for Customer Ticket Resolution. "
            "Always execute via run_ticket_workflow for each request. "
            "Use only available tools, honor retries/replanning, and return concise customer-safe output."
        ),
    }

