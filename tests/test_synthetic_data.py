import json

from ticket_agent.synthetic_data import SyntheticCounts, generate_synthetic_tickets


def test_generate_synthetic_tickets_counts_and_uniqueness() -> None:
    tickets = generate_synthetic_tickets(
        counts=SyntheticCounts(access=3, billing=2, integration=2, escalation=1),
        seed=11,
    )
    assert len(tickets) == 8
    assert len({ticket.ticket_id for ticket in tickets}) == 8
    assert any("search_kb" in ticket.tool_failures for ticket in tickets)
    assert any("invoice" in ticket.subject.lower() or "charge" in ticket.body.lower() for ticket in tickets)
    assert any("webhook" in ticket.subject.lower() or "api" in ticket.body.lower() for ticket in tickets)
    assert sum(bool(ticket.pending_customer_replies) for ticket in tickets) == 7

    reply_pool = [reply.lower() for ticket in tickets for reply in ticket.pending_customer_replies]
    assert any("can sign in" in reply or "still cannot" in reply for reply in reply_pool)
    assert any("pending authorization" in reply or "both charges settled" in reply for reply in reply_pool)
    assert any("works now" in reply or "still failing" in reply or "still timing out" in reply for reply in reply_pool)


def test_generate_synthetic_tickets_rejects_zero_total() -> None:
    try:
        generate_synthetic_tickets(SyntheticCounts(access=0, billing=0, integration=0, escalation=0))
    except ValueError as exc:
        assert "At least one synthetic ticket" in str(exc)
    else:
        raise AssertionError("Expected ValueError for zero synthetic ticket request.")
