from __future__ import annotations

from ticket_agent.cli import build_agent
from ticket_agent.config import Settings


class TicketSupportToolset:
    """A2A toolset that wraps the existing ticket-agent runtime."""

    def __init__(self) -> None:
        self._agent = build_agent(Settings())

    async def resolve_ticket(self, ticket_id: str, goal: str | None = None) -> dict[str, object]:
        """Resolve one ticket and return agent output.

        Args:
            ticket_id: Ticket ID (for example, TICK-1001).
            goal: Optional custom goal for the run.

        Returns:
            Execution outcome and final response metadata.
        """
        run_goal = goal or "Resolve the customer ticket end-to-end using only the approved tools."
        log = self._agent.run(ticket_id=ticket_id, goal=run_goal)
        log_path = self._agent.save_log(log)
        return {
            "ticket_id": ticket_id,
            "status": log.status,
            "final_ticket_status": log.final_ticket_status,
            "final_response": log.final_response,
            "response_source": log.response_source,
            "log_path": str(log_path),
        }

    def get_tools(self) -> dict[str, object]:
        return {
            "resolve_ticket": self,
        }

