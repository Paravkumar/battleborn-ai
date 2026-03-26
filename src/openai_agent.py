from __future__ import annotations

from .agent_toolset import SupportToolset


def create_agent() -> dict[str, object]:
    """Create cloud-capable support agent definition."""
    toolset = SupportToolset()
    return {
        "tools": toolset.get_tools(),
        "system_prompt": (
            "You are Battleborn Customer Support AI. "
            "Resolve requests directly, use RAG context, avoid unnecessary escalation."
        ),
    }

