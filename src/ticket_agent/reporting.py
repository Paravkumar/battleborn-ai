from __future__ import annotations

from pathlib import Path

from ticket_agent.dataset import load_log
from ticket_agent.schemas import ExecutionLog


def _format_datetime(value) -> str:
    if value is None:
        return "n/a"
    return value.isoformat().replace("+00:00", "Z")


def _format_datetime_human(value) -> str:
    if value is None:
        return "n/a"
    return value.strftime("%Y-%m-%d %H:%M:%S UTC")


def _retry_count(log: ExecutionLog) -> int:
    return sum(1 for step in log.steps if step.attempt > 1 and step.outcome != "skipped")


def _compact_text(text: str, max_length: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3].rstrip() + "..."


def _grounding_hits(log: ExecutionLog) -> list[dict[str, object]]:
    deduped: dict[str, dict[str, object]] = {}
    for step in log.steps:
        if step.tool != "search_kb" or step.outcome != "success":
            continue
        hits = step.data.get("hits", [])
        if not isinstance(hits, list):
            continue
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            article_id = hit.get("article_id")
            if isinstance(article_id, str) and article_id not in deduped:
                deduped[article_id] = hit
    return list(deduped.values())


def build_execution_log_report(log: ExecutionLog) -> str:
    ticket = log.ticket_snapshot
    hits = _grounding_hits(log)
    lines = [
        f"# Ticket Report: {log.ticket_id}",
        "",
        f"- Status: `{log.status}`",
        f"- Final ticket status: `{log.final_ticket_status or 'unknown'}`",
        f"- Plan source: `{log.plan_source}`",
        f"- Response source: `{log.response_source}`",
        f"- Model: `{log.model_used or 'none'}`",
        f"- Retries observed: `{_retry_count(log)}`",
        f"- Started: `{_format_datetime(log.started_at)}`",
        f"- Finished: `{_format_datetime(log.finished_at)}`",
        "",
        "## Ticket",
        "",
        f"- Subject: {ticket.get('subject', '')}",
        f"- Priority: `{ticket.get('priority', 'unknown')}`",
        f"- Goal: {log.goal}",
        "",
        "## Plan",
        "",
    ]
    for index, step in enumerate(log.plan.steps, start=1):
        lines.append(f"{index}. `{step.tool}`: {step.purpose}")

    lines.extend(["", "## Execution Timeline", ""])
    for step in log.steps:
        lines.append(
            f"- `{step.step_id}` `{step.tool}` attempt `{step.attempt}` -> `{step.outcome}`: {step.message}"
        )

    lines.extend(["", "## Grounding", ""])
    if hits:
        for hit in hits:
            matched_terms = ", ".join(hit.get("matched_terms", [])) or "none"
            confidence = hit.get("confidence_score", "n/a")
            lines.append(
                f"- `{hit.get('article_id', 'unknown')}` {hit.get('title', '')} | type `{hit.get('information_type', 'unknown')}` | confidence `{confidence}` | matched terms: {matched_terms}"
            )
    else:
        lines.append("- No successful KB grounding hits were recorded.")

    lines.extend(["", "## Outcome", ""])
    if log.final_response:
        lines.append(log.final_response)
    else:
        lines.append("No final response recorded.")

    return "\n".join(lines).strip() + "\n"


def build_demo_report(logs: list[ExecutionLog]) -> str:
    ordered_logs = sorted(logs, key=lambda item: item.started_at)
    lines = [
        "# Ticket Agent Demo Report",
        "",
        f"Generated from `{len(ordered_logs)}` execution logs.",
        "",
        "| Ticket | Status | Plan | Response | Retries | Final Ticket Status |",
        "|---|---|---|---|---:|---|",
    ]

    for log in ordered_logs:
        lines.append(
            f"| {log.ticket_id} | {log.status} | {log.plan_source} | {log.response_source} | {_retry_count(log)} | {log.final_ticket_status or 'unknown'} |"
        )

    lines.extend(["", "## Scenario Highlights", ""])
    for log in ordered_logs:
        ticket = log.ticket_snapshot
        lines.append(f"### {log.ticket_id}")
        lines.append("")
        lines.append(f"- Subject: {ticket.get('subject', '')}")
        lines.append(f"- Outcome: `{log.status}` with final ticket status `{log.final_ticket_status or 'unknown'}`")
        hits = _grounding_hits(log)
        if hits:
            top_hit = hits[0]
            matched_terms = ", ".join(top_hit.get("matched_terms", [])) or "none"
            lines.append(
                f"- Grounding: `{top_hit.get('article_id', 'unknown')}` matched on {matched_terms} with confidence `{top_hit.get('confidence_score', 'n/a')}`"
            )
        else:
            lines.append("- Grounding: no successful KB evidence; workflow escalated or failed before resolution.")
        if log.final_response:
            lines.append(f"- Final response: {_compact_text(log.final_response)}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def latest_unique_log_paths(logs_dir: Path, latest: int = 3) -> list[Path]:
    paths = sorted(logs_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    selected: list[Path] = []
    seen_ticket_ids: set[str] = set()
    for path in paths:
        try:
            log = load_log(path)
        except Exception:
            continue
        if log.ticket_id in seen_ticket_ids:
            continue
        selected.append(path)
        seen_ticket_ids.add(log.ticket_id)
        if len(selected) >= latest:
            break
    return selected


def build_ticket_summary(log: ExecutionLog) -> str:
    """Generate a comprehensive human-readable summary for a ticket."""
    ticket = log.ticket_snapshot
    hits = _grounding_hits(log)
    
    # Extract key information
    ticket_id = log.ticket_id
    subject = ticket.get("subject", "No subject")
    body = ticket.get("body", "No description")
    priority = ticket.get("priority", "unknown")
    issue_type = ticket.get("issue_type") or "Not classified"
    status = log.status
    final_status = log.final_ticket_status or "unknown"
    
    # Get classification info from steps
    classification_info = None
    for step in log.steps:
        if step.tool == "classify_ticket" and step.outcome == "success":
            classification_info = step.data
            break
    
    # Get response history
    agent_responses = []
    customer_messages = [body]  # Start with initial body
    for step in log.steps:
        if step.tool == "post_response" and step.outcome == "success":
            agent_responses.append(step.arguments.get("response_body", ""))
        if step.tool == "customer_reply" and step.outcome == "success":
            customer_messages.append(step.data.get("message", ""))
        if step.tool == "escalate_ticket" and step.outcome == "escalated":
            agent_responses.append(step.data.get("customer_message", ""))
    
    # Build summary sections
    lines = [
        "=" * 70,
        f"TICKET SUMMARY: {ticket_id}",
        "=" * 70,
        "",
        "## TICKET INFORMATION",
        f"  Subject:    {subject}",
        f"  Priority:   {priority.upper()}",
        f"  Issue Type: {issue_type}",
        f"  Status:     {final_status.upper()}",
        f"  Outcome:    {status.upper()}",
        "",
    ]
    
    # Classification details
    if classification_info:
        lines.extend([
            "## CLASSIFICATION",
            f"  Issue Type:  {classification_info.get('issue_type', 'unknown')}",
            f"  Intent:      {classification_info.get('intent', 'unknown')}",
            f"  Confidence:  {classification_info.get('confidence', 'unknown')} ({classification_info.get('confidence_score', 0):.0%})",
            f"  Priority:    {classification_info.get('priority', priority)}",
            "",
        ])
    
    # Customer's request
    lines.extend([
        "## CUSTOMER REQUEST",
        f"  {_compact_text(body, max_length=500)}",
        "",
    ])
    
    # Conversation history
    if len(customer_messages) > 1 or agent_responses:
        lines.append("## CONVERSATION HISTORY")
        msg_index = 0
        resp_index = 0
        
        # Interleave messages (customer first, then agent response)
        while msg_index < len(customer_messages) or resp_index < len(agent_responses):
            if msg_index < len(customer_messages):
                prefix = "Initial Request" if msg_index == 0 else f"Customer Reply {msg_index}"
                lines.append(f"  [{prefix}]")
                lines.append(f"    {_compact_text(customer_messages[msg_index], max_length=300)}")
                lines.append("")
                msg_index += 1
            
            if resp_index < len(agent_responses):
                lines.append(f"  [Agent Response {resp_index + 1}]")
                lines.append(f"    {_compact_text(agent_responses[resp_index], max_length=300)}")
                lines.append("")
                resp_index += 1
    
    # KB Articles used
    if hits:
        lines.append("## KNOWLEDGE BASE ARTICLES USED")
        for hit in hits:
            lines.append(f"  - {hit.get('article_id', 'unknown')}: {hit.get('title', 'No title')}")
            lines.append(f"    Type: {hit.get('information_type', 'unknown')} | Confidence: {hit.get('confidence_score', 0):.0%}")
        lines.append("")
    
    # Resolution or Escalation details
    if status == "completed":
        lines.extend([
            "## RESOLUTION",
            "  Status: RESOLVED",
            f"  The ticket was successfully resolved by the automated support agent.",
        ])
        if agent_responses:
            lines.append(f"  Final Response: {_compact_text(agent_responses[-1], max_length=200)}")
    elif status == "escalated":
        lines.extend([
            "## ESCALATION",
            "  Status: ESCALATED TO HUMAN SUPPORT",
        ])
        # Find escalation reason
        escalation_reason = None
        for step in log.steps:
            if step.tool == "escalate_ticket":
                escalation_reason = step.arguments.get("reason", "")
                break
        if escalation_reason:
            # Parse the escalation reason for key info
            lines.append("  Reason for Escalation:")
            if "Classification:" in escalation_reason:
                lines.append(f"    {_compact_text(escalation_reason, max_length=400)}")
            else:
                lines.append(f"    {_compact_text(escalation_reason, max_length=300)}")
        if log.final_response:
            lines.append(f"  Escalation Summary: {_compact_text(log.final_response, max_length=300)}")
    elif status == "waiting_on_customer":
        lines.extend([
            "## STATUS: AWAITING CUSTOMER RESPONSE",
            "  The agent is waiting for additional information from the customer.",
        ])
        if agent_responses:
            lines.append(f"  Last Agent Message: {_compact_text(agent_responses[-1], max_length=200)}")
    elif status == "failed":
        lines.extend([
            "## STATUS: FAILED",
            "  The automated resolution could not be completed.",
        ])
        if log.final_response:
            lines.append(f"  Details: {_compact_text(log.final_response, max_length=300)}")
    
    lines.extend([
        "",
        "## PROCESSING DETAILS",
        f"  Started:  {_format_datetime_human(log.started_at)}",
        f"  Finished: {_format_datetime_human(log.finished_at)}",
        f"  Model:    {log.model_used or 'fallback (no LLM)'}",
        f"  Retries:  {_retry_count(log)}",
        "",
        "=" * 70,
    ])
    
    return "\n".join(lines)


def build_all_tickets_summary(logs: list[ExecutionLog]) -> str:
    """Generate a summary report for multiple tickets."""
    ordered_logs = sorted(logs, key=lambda item: item.started_at)
    
    # Statistics
    total = len(ordered_logs)
    resolved = sum(1 for log in ordered_logs if log.status == "completed")
    escalated = sum(1 for log in ordered_logs if log.status == "escalated")
    waiting = sum(1 for log in ordered_logs if log.status == "waiting_on_customer")
    failed = sum(1 for log in ordered_logs if log.status == "failed")
    
    lines = [
        "=" * 70,
        "TICKET SUMMARY REPORT",
        "=" * 70,
        "",
        "## OVERVIEW",
        f"  Total Tickets:     {total}",
        f"  Resolved:          {resolved} ({resolved/total*100:.0f}%)" if total > 0 else "  Resolved:          0",
        f"  Escalated:         {escalated} ({escalated/total*100:.0f}%)" if total > 0 else "  Escalated:         0",
        f"  Awaiting Customer: {waiting}",
        f"  Failed:            {failed}",
        "",
        "-" * 70,
        "",
    ]
    
    # Individual ticket summaries
    for log in ordered_logs:
        lines.append(build_ticket_summary(log))
        lines.append("")
    
    return "\n".join(lines)
