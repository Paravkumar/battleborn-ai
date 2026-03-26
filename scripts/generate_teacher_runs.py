from __future__ import annotations

import argparse
import json
from pathlib import Path

from ticket_agent.cli import build_agent
from ticket_agent.config import Settings
from ticket_agent.dataset import execution_log_to_sft_example
from ticket_agent.repository import TicketRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate teacher traces and export them as SFT examples.")
    parser.add_argument("--ticket-id", action="append", dest="ticket_ids", default=[])
    parser.add_argument("--all-tickets", action="store_true")
    parser.add_argument("--model", default="llama3.1:70b")
    parser.add_argument("--tickets-file", type=Path, default=Path("data/tickets.json"))
    parser.add_argument(
        "--goal",
        default="Resolve the customer ticket end-to-end using only the approved tools.",
    )
    parser.add_argument("--output-jsonl", type=Path, default=Path("artifacts/teacher_runs.jsonl"))
    parser.add_argument("--require-ollama-plan", action="store_true")
    parser.add_argument("--require-ollama-draft", action="store_true")
    return parser


def resolve_ticket_ids(repository: TicketRepository, args: argparse.Namespace) -> list[str]:
    if args.all_tickets:
        return repository.list_ticket_ids()
    if args.ticket_ids:
        return args.ticket_ids
    raise SystemExit("Pass --all-tickets or at least one --ticket-id.")


def main() -> int:
    args = build_parser().parse_args()
    settings = Settings(use_ollama=True, student_model=args.model)
    repository = TicketRepository(args.tickets_file)
    ticket_ids = resolve_ticket_ids(repository, args)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for ticket_id in ticket_ids:
            agent = build_agent(settings, tickets_path=args.tickets_file)
            log = agent.run(ticket_id=ticket_id, goal=args.goal)
            path = agent.save_log(log)

            if log.status == "waiting_on_customer":
                skipped += 1
                print(f"Skipped {ticket_id}: workflow is still waiting on customer input. Log saved to {path}")
                continue
            if args.require_ollama_plan and log.plan_source != "ollama":
                skipped += 1
                print(f"Skipped {ticket_id}: planner fell back. Log saved to {path}")
                continue
            escalation_without_draft = log.status == "escalated" and log.response_source == "none"
            if args.require_ollama_draft and not escalation_without_draft and log.response_source != "ollama":
                skipped += 1
                print(f"Skipped {ticket_id}: draft fell back. Log saved to {path}")
                continue

            example = execution_log_to_sft_example(log)
            handle.write(json.dumps(example))
            handle.write("\n")
            written += 1
            print(
                f"Captured {ticket_id}: status={log.status}, plan_source={log.plan_source}, "
                f"response_source={log.response_source}, log={path}"
            )

    print(f"\nWrote {written} examples to {args.output_jsonl}. Skipped {skipped}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
