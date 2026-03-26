from __future__ import annotations

import argparse
import json
from pathlib import Path

from ticket_agent.synthetic_data import SyntheticCounts, generate_synthetic_tickets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic tickets for teacher-trace collection.")
    parser.add_argument("--access-count", type=int, default=25)
    parser.add_argument("--billing-count", type=int, default=25)
    parser.add_argument("--integration-count", type=int, default=25)
    parser.add_argument("--escalation-count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("artifacts/generated_tickets.json"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    counts = SyntheticCounts(
        access=args.access_count,
        billing=args.billing_count,
        integration=args.integration_count,
        escalation=args.escalation_count,
    )
    tickets = generate_synthetic_tickets(counts=counts, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([ticket.model_dump(mode="json") for ticket in tickets], indent=2),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(tickets)} tickets to {args.output} "
        f"(access={counts.access}, billing={counts.billing}, integration={counts.integration}, escalation={counts.escalation})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
