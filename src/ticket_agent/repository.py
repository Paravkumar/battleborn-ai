from __future__ import annotations

import json
import re
from pathlib import Path

from ticket_agent.schemas import Ticket


class TicketRepository:
    def __init__(self, tickets_path: Path) -> None:
        self.tickets_path = tickets_path
        if tickets_path.exists():
            raw = json.loads(tickets_path.read_text(encoding="utf-8"))
        else:
            raw = []
        self._tickets = {
            item["ticket_id"]: Ticket.model_validate(item)
            for item in raw
        }

    def get(self, ticket_id: str) -> Ticket:
        try:
            return self._tickets[ticket_id]
        except KeyError as exc:
            raise KeyError(f"Unknown ticket_id: {ticket_id}") from exc

    def list_ticket_ids(self) -> list[str]:
        return list(self._tickets.keys())

    def create_ticket(
        self,
        subject: str,
        body: str,
        priority: str = "medium",
        customer_id: str | None = None,
        ticket_id: str | None = None,
        pending_customer_replies: list[str] | None = None,
    ) -> Ticket:
        resolved_ticket_id = ticket_id or self._next_ticket_id()
        if resolved_ticket_id in self._tickets:
            raise ValueError(f"Ticket already exists: {resolved_ticket_id}")

        resolved_customer_id = customer_id or f"CUST-{resolved_ticket_id}"
        ticket = Ticket(
            ticket_id=resolved_ticket_id,
            customer_id=resolved_customer_id,
            subject=subject.strip(),
            body=body.strip(),
            priority=priority,
            pending_customer_replies=list(pending_customer_replies or []),
        )
        self._tickets[resolved_ticket_id] = ticket
        self.save()
        return ticket

    def save(self) -> None:
        self.tickets_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [ticket.model_dump(mode="json") for ticket in self._tickets.values()]
        self.tickets_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def append_response(self, ticket_id: str, response: str) -> Ticket:
        ticket = self.get(ticket_id)
        ticket.response_history.append(response)
        self.save()
        return ticket

    def customer_messages(self, ticket_id: str) -> list[str]:
        ticket = self.get(ticket_id)
        return [ticket.body, *ticket.customer_message_history]

    def current_customer_message(self, ticket_id: str) -> str:
        return self.customer_messages(ticket_id)[-1]

    def consume_next_customer_reply(self, ticket_id: str) -> str | None:
        ticket = self.get(ticket_id)
        if not ticket.pending_customer_replies:
            return None
        message = ticket.pending_customer_replies.pop(0)
        ticket.customer_message_history.append(message)
        ticket.status = "open"
        ticket.resolution_summary = None
        self.save()
        return message

    def add_customer_message(self, ticket_id: str, message: str) -> Ticket:
        ticket = self.get(ticket_id)
        ticket.customer_message_history.append(message)
        ticket.status = "open"
        ticket.resolution_summary = None
        self.save()
        return ticket

    def clear_pending_customer_replies(self, ticket_id: str) -> Ticket:
        ticket = self.get(ticket_id)
        ticket.pending_customer_replies = []
        self.save()
        return ticket

    def update_status(self, ticket_id: str, status: str, resolution_summary: str | None = None) -> Ticket:
        ticket = self.get(ticket_id)
        ticket.status = status
        ticket.resolution_summary = resolution_summary
        self.save()
        return ticket

    def set_issue_type(self, ticket_id: str, issue_type: str) -> Ticket:
        ticket = self.get(ticket_id)
        ticket.issue_type = issue_type
        self.save()
        return ticket

    def set_priority(self, ticket_id: str, priority: str) -> Ticket:
        ticket = self.get(ticket_id)
        ticket.priority = priority
        self.save()
        return ticket

    def consume_tool_failure(self, ticket_id: str, tool_name: str) -> bool:
        ticket = self.get(ticket_id)
        remaining = ticket.tool_failures.get(tool_name, 0)
        if remaining <= 0:
            return False
        ticket.tool_failures[tool_name] = remaining - 1
        self.save()
        return True

    def _next_ticket_id(self) -> str:
        matches: list[tuple[str, int, int]] = []
        for ticket_id in self._tickets:
            match = re.fullmatch(r"([A-Z]+)-(\d+)", ticket_id)
            if match is None:
                continue
            prefix, raw_number = match.groups()
            matches.append((prefix, int(raw_number), len(raw_number)))

        if not matches:
            return "TICK-1001"

        preferred_prefix = "TICK" if any(prefix == "TICK" for prefix, _, _ in matches) else matches[0][0]
        prefix_matches = [(number, width) for prefix, number, width in matches if prefix == preferred_prefix]
        next_number = max(number for number, _ in prefix_matches) + 1
        width = max(width for _, width in prefix_matches)
        return f"{preferred_prefix}-{next_number:0{width}d}"
