from __future__ import annotations

from a2a_agent.agent_toolset import TicketSupportToolset


def create_agent() -> dict[str, object]:
    """Create OpenAI agent configuration and tools."""
    toolset = TicketSupportToolset()
    return {
        "tools": toolset.get_tools(),
        "system_prompt": (
            "You are a customer-support orchestration agent for Battleborn. "
            "Use resolve_ticket when a ticket_id is provided, then summarize the result clearly."
        ),
    }

