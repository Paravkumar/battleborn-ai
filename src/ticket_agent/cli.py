from __future__ import annotations

import argparse
import json
from pathlib import Path

from ticket_agent.agent import TicketResolutionAgent
from ticket_agent.config import Settings
from ticket_agent.dataset import execution_log_to_sft_example, load_log
from ticket_agent.few_shot import load_response_examples
from ticket_agent.kb_ingest import load_kb_articles_from_docs
from ticket_agent.knowledge_base import KnowledgeBase
from ticket_agent.ollama_client import OllamaGateway
from ticket_agent.reporting import build_demo_report, build_execution_log_report, build_ticket_summary, build_all_tickets_summary, latest_unique_log_paths
from ticket_agent.repository import TicketRepository
from ticket_agent.schemas import ExecutionLog, KnowledgeBaseArticle
from ticket_agent.tools import TicketResolutionToolkit


def build_agent(settings: Settings, tickets_path: Path | None = None) -> TicketResolutionAgent:
    repository = TicketRepository(tickets_path or settings.data_dir / "tickets.json")
    kb = KnowledgeBase(settings.data_dir / "knowledge_base.json")
    llm = OllamaGateway(settings.ollama_host) if settings.use_ollama else None
    response_examples = load_response_examples(settings.teacher_examples_path, limit=settings.max_response_examples)
    toolkit = TicketResolutionToolkit(
        repository=repository,
        kb=kb,
        llm=llm,
        student_model=settings.student_model,
        use_ollama=settings.use_ollama,
        min_kb_confidence=settings.min_kb_confidence,
        min_classification_confidence=settings.min_classification_confidence,
        response_examples=response_examples,
    )
    return TicketResolutionAgent(
        settings=settings,
        repository=repository,
        toolkit=toolkit,
        llm=llm,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Customer ticket resolution agent MVP.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create-ticket", help="Create a new ticket in the JSON ticket store.")
    create_parser.add_argument("--subject", default=None)
    create_parser.add_argument("--body", default=None)
    create_parser.add_argument("--customer-id", default=None)
    create_parser.add_argument("--ticket-id", default=None)
    create_parser.add_argument(
        "--priority",
        choices=["low", "medium", "high", "urgent"],
        default="medium",
    )
    create_parser.add_argument(
        "--scripted-reply",
        action="append",
        default=[],
        help="Optional scripted customer follow-up reply. Pass multiple times to queue several replies.",
    )
    create_parser.add_argument("--tickets-file", type=Path, default=None)

    import_kb_parser = subparsers.add_parser("import-kb", help="Import KB articles from a JSON file into the expandable KB store.")
    import_kb_parser.add_argument("--source", required=True, type=Path)
    import_kb_parser.add_argument("--target", type=Path, default=None)
    import_kb_parser.add_argument("--replace-existing", action="store_true")

    import_kb_docs_parser = subparsers.add_parser(
        "import-kb-docs",
        help="Import markdown or text docs and convert them into KB articles in the expandable KB store.",
    )
    import_kb_docs_parser.add_argument("--source", required=True, type=Path)
    import_kb_docs_parser.add_argument(
        "--issue-type",
        choices=["access_management", "billing", "integration", "general"],
        default="general",
    )
    import_kb_docs_parser.add_argument(
        "--information-type",
        choices=["policy", "product", "hybrid"],
        default="hybrid",
    )
    import_kb_docs_parser.add_argument("--article-prefix", default="KB-DOC")
    import_kb_docs_parser.add_argument("--target", type=Path, default=None)
    import_kb_docs_parser.add_argument("--replace-existing", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run the agent on a sample ticket.")
    run_parser.add_argument("--ticket-id", required=True)
    run_parser.add_argument(
        "--goal",
        default="Resolve the customer ticket end-to-end using only the approved tools.",
    )
    run_parser.add_argument("--disable-ollama", action="store_true")
    run_parser.add_argument("--student-model", default=None)
    run_parser.add_argument("--tickets-file", type=Path, default=None)

    chat_parser = subparsers.add_parser("chat", help="Run the agent in an interactive multi-turn chat loop.")
    chat_parser.add_argument("--ticket-id", required=True)
    chat_parser.add_argument(
        "--goal",
        default="Resolve the customer ticket end-to-end using only the approved tools.",
    )
    chat_parser.add_argument("--disable-ollama", action="store_true")
    chat_parser.add_argument("--student-model", default=None)
    chat_parser.add_argument("--tickets-file", type=Path, default=None)
    chat_parser.add_argument(
        "--use-scripted-replies",
        action="store_true",
        help="Keep any pending scripted customer replies from the ticket fixture instead of waiting for live input.",
    )

    export_parser = subparsers.add_parser("export-sft", help="Convert an execution log into one SFT example.")
    export_parser.add_argument("--log", required=True, type=Path)
    export_parser.add_argument("--out", required=False, type=Path)

    summarize_log_parser = subparsers.add_parser("summarize-log", help="Render one execution log as a markdown report.")
    summarize_log_parser.add_argument("--log", required=True, type=Path)
    summarize_log_parser.add_argument("--out", required=False, type=Path)

    summarize_demo_parser = subparsers.add_parser("summarize-demo", help="Render a demo report from recent execution logs.")
    summarize_demo_parser.add_argument("--logs", nargs="*", type=Path, default=None)
    summarize_demo_parser.add_argument("--logs-dir", required=False, type=Path, default=None)
    summarize_demo_parser.add_argument("--latest", type=int, default=3)
    summarize_demo_parser.add_argument("--out", required=False, type=Path)

    summary_parser = subparsers.add_parser("summary", help="Generate a comprehensive ticket summary from execution logs.")
    summary_parser.add_argument("--log", required=False, type=Path, help="Single execution log file to summarize.")
    summary_parser.add_argument("--logs", nargs="*", type=Path, default=None, help="Multiple log files to summarize.")
    summary_parser.add_argument("--logs-dir", required=False, type=Path, default=None, help="Directory containing log files.")
    summary_parser.add_argument("--latest", type=int, default=10, help="Number of latest logs to include when using --logs-dir.")
    summary_parser.add_argument("--out", required=False, type=Path, help="Output file path for the summary.")

    return parser


def _print_chat_transcript(log: ExecutionLog, include_initial_customer: bool = False) -> None:
    if include_initial_customer:
        initial_message = str(log.ticket_snapshot.get("body", "")).strip()
        if initial_message:
            print(f"Customer> {initial_message}\n")

    for step in log.steps:
        if step.tool == "customer_reply" and step.outcome == "success":
            customer_message = str(step.data.get("message", "")).strip()
            if customer_message:
                print(f"Customer> {customer_message}\n")
        if step.tool == "post_response" and step.outcome == "success":
            agent_message = str(step.arguments.get("response_body", "")).strip()
            if agent_message:
                print(f"Agent> {agent_message}\n")
        if step.tool == "escalate_ticket" and step.outcome == "escalated":
            agent_message = str(step.data.get("customer_message", "")).strip()
            if agent_message:
                print(f"Agent> {agent_message}\n")


def _prompt_required(value: str | None, label: str) -> str:
    if value is not None and value.strip():
        return value.strip()
    while True:
        entered = input(f"{label}> ").strip()
        if entered:
            return entered
        print(f"{label} is required.")


def _load_kb_article_payload(path: Path) -> list[KnowledgeBaseArticle]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise SystemExit(f"KB source must be a JSON object or list of objects: {path}")
    return [KnowledgeBaseArticle.model_validate(item) for item in payload]


def _merge_kb_articles(
    existing: list[KnowledgeBaseArticle],
    incoming: list[KnowledgeBaseArticle],
    replace_existing: bool,
) -> tuple[list[KnowledgeBaseArticle], int, int]:
    merged: dict[str, KnowledgeBaseArticle] = {article.article_id: article for article in existing}
    added = 0
    replaced = 0
    for article in incoming:
        if article.article_id in merged:
            if not replace_existing:
                continue
            replaced += 1
        else:
            added += 1
        merged[article.article_id] = article
    ordered = sorted(merged.values(), key=lambda article: article.article_id)
    return ordered, added, replaced


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "create-ticket":
        settings = Settings()
        repository = TicketRepository(args.tickets_file or settings.data_dir / "tickets.json")
        subject = _prompt_required(args.subject, "Subject")
        body = _prompt_required(args.body, "Body")
        ticket = repository.create_ticket(
            subject=subject,
            body=body,
            priority=args.priority,
            customer_id=args.customer_id,
            ticket_id=args.ticket_id,
            pending_customer_replies=args.scripted_reply,
        )
        print(json.dumps(ticket.model_dump(mode="json"), indent=2))
        print(
            f"\nCreated ticket {ticket.ticket_id} in {repository.tickets_path}\n"
            f"Run it with: ticket-agent chat --ticket-id {ticket.ticket_id}"
        )
        return 0

    if args.command == "import-kb":
        settings = Settings()
        target = args.target or (settings.data_dir / "knowledge_base_extra.json")
        source_articles = _load_kb_article_payload(args.source)
        base_path = settings.data_dir / "knowledge_base.json"
        if target.resolve() != base_path.resolve() and base_path.exists():
            base_articles = _load_kb_article_payload(base_path)
            base_ids = {article.article_id for article in base_articles}
            colliding_ids = sorted(article.article_id for article in source_articles if article.article_id in base_ids)
            if colliding_ids:
                raise SystemExit(
                    "KB import would collide with curated base article IDs: "
                    + ", ".join(colliding_ids)
                )
        existing_articles = []
        if target.exists():
            existing_articles = _load_kb_article_payload(target)
        merged_articles, added, replaced = _merge_kb_articles(
            existing=existing_articles,
            incoming=source_articles,
            replace_existing=args.replace_existing,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps([article.model_dump(mode="json") for article in merged_articles], indent=2),
            encoding="utf-8",
        )
        kb = KnowledgeBase(settings.data_dir / "knowledge_base.json")
        print(
            f"Imported {added + replaced} article(s) into {target} "
            f"(added={added}, replaced={replaced}, loaded_total={kb.article_count()})."
        )
        return 0

    if args.command == "import-kb-docs":
        settings = Settings()
        target = args.target or (settings.data_dir / "knowledge_base_extra.json")
        source_articles = load_kb_articles_from_docs(
            source=args.source,
            issue_type=args.issue_type,
            information_type=args.information_type,
            article_prefix=args.article_prefix,
        )
        base_path = settings.data_dir / "knowledge_base.json"
        if target.resolve() != base_path.resolve() and base_path.exists():
            base_articles = _load_kb_article_payload(base_path)
            base_ids = {article.article_id for article in base_articles}
            colliding_ids = sorted(article.article_id for article in source_articles if article.article_id in base_ids)
            if colliding_ids:
                raise SystemExit(
                    "KB import would collide with curated base article IDs: "
                    + ", ".join(colliding_ids)
                )
        existing_articles = []
        if target.exists():
            existing_articles = _load_kb_article_payload(target)
        merged_articles, added, replaced = _merge_kb_articles(
            existing=existing_articles,
            incoming=source_articles,
            replace_existing=args.replace_existing,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps([article.model_dump(mode="json") for article in merged_articles], indent=2),
            encoding="utf-8",
        )
        kb = KnowledgeBase(settings.data_dir / "knowledge_base.json")
        print(
            f"Imported {added + replaced} article(s) from docs into {target} "
            f"(added={added}, replaced={replaced}, loaded_total={kb.article_count()})."
        )
        return 0

    if args.command == "run":
        settings = Settings(use_ollama=not args.disable_ollama)
        if args.student_model:
            settings.student_model = args.student_model
        agent = build_agent(settings, tickets_path=args.tickets_file)
        log = agent.run(ticket_id=args.ticket_id, goal=args.goal)
        path = agent.save_log(log)
        print(json.dumps(log.model_dump(mode="json"), indent=2))
        print(f"\nSaved execution log to: {path}")
        return 0

    if args.command == "chat":
        settings = Settings(use_ollama=not args.disable_ollama)
        if args.student_model:
            settings.student_model = args.student_model
        agent = build_agent(settings, tickets_path=args.tickets_file)
        if not args.use_scripted_replies:
            agent.repository.clear_pending_customer_replies(args.ticket_id)

        ticket = agent.repository.get(args.ticket_id)
        print(f"Ticket {ticket.ticket_id}: {ticket.subject}\n")

        include_initial_customer = True
        while True:
            log = agent.run(ticket_id=args.ticket_id, goal=args.goal)
            path = agent.save_log(log)

            _print_chat_transcript(log, include_initial_customer=include_initial_customer)
            include_initial_customer = False

            if log.status in {"completed", "escalated", "failed"}:
                print(f"Conversation ended with status: {log.status}")
                print(f"Saved execution log to: {path}")
                return 0
            if args.use_scripted_replies and log.status == "waiting_on_customer":
                print("Conversation paused: no scripted customer reply remains.")
                print(f"Saved execution log to: {path}")
                return 0
            if log.status != "waiting_on_customer":
                return 1

            try:
                customer_reply = input("Customer> ").strip()
            except EOFError:
                return 0
            if not customer_reply:
                print("Ending chat without another customer reply.")
                return 0
            agent.repository.add_customer_message(args.ticket_id, customer_reply)

    if args.command == "export-sft":
        log = load_log(args.log)
        example = execution_log_to_sft_example(log)
        payload = json.dumps(example, indent=2)
        if args.out:
            args.out.write_text(payload, encoding="utf-8")
            print(f"Wrote SFT example to: {args.out}")
        else:
            print(payload)
        return 0

    if args.command == "summarize-log":
        log = load_log(args.log)
        report = build_execution_log_report(log)
        if args.out:
            args.out.write_text(report, encoding="utf-8")
            print(f"Wrote markdown report to: {args.out}")
        else:
            print(report)
        return 0

    if args.command == "summarize-demo":
        settings = Settings(use_ollama=False)
        log_paths = args.logs
        if not log_paths:
            logs_dir = args.logs_dir or settings.output_dir
            log_paths = latest_unique_log_paths(logs_dir, latest=args.latest)
        logs = [load_log(path) for path in log_paths]
        report = build_demo_report(logs)
        if args.out:
            args.out.write_text(report, encoding="utf-8")
            print(f"Wrote demo report to: {args.out}")
        else:
            print(report)
        return 0

    if args.command == "summary":
        settings = Settings(use_ollama=False)
        
        if args.log:
            # Single log summary
            log = load_log(args.log)
            report = build_ticket_summary(log)
        else:
            # Multiple logs summary
            log_paths = args.logs
            if not log_paths:
                logs_dir = args.logs_dir or settings.output_dir
                log_paths = latest_unique_log_paths(logs_dir, latest=args.latest)
            
            if not log_paths:
                print("No execution logs found to summarize.")
                return 1
            
            logs = [load_log(path) for path in log_paths]
            report = build_all_tickets_summary(logs)
        
        if args.out:
            args.out.write_text(report, encoding="utf-8")
            print(f"Wrote ticket summary to: {args.out}")
        else:
            print(report)
        return 0

    parser.print_help()
    return 1
