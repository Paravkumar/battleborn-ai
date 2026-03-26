import json
import sys
from pathlib import Path

from ticket_agent.cli import main
from ticket_agent.repository import TicketRepository


def test_repository_create_ticket_persists_and_generates_next_id(tmp_path: Path) -> None:
    tickets_path = tmp_path / "tickets.json"
    tickets_path.write_text(
        json.dumps(
            [
                {
                    "ticket_id": "TICK-1001",
                    "customer_id": "CUST-001",
                    "subject": "Existing ticket",
                    "body": "Existing body",
                    "priority": "medium",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    repository = TicketRepository(tickets_path)
    created = repository.create_ticket(
        subject="New issue",
        body="Please help with a new issue.",
        priority="high",
        pending_customer_replies=["It still is not fixed."],
    )

    assert created.ticket_id == "TICK-1002"
    assert created.customer_id == "CUST-TICK-1002"

    reloaded = TicketRepository(tickets_path)
    persisted = reloaded.get("TICK-1002")
    assert persisted.subject == "New issue"
    assert persisted.priority == "high"
    assert persisted.pending_customer_replies == ["It still is not fixed."]


def test_create_ticket_command_writes_ticket_file(tmp_path: Path, monkeypatch, capsys) -> None:
    tickets_path = tmp_path / "tickets.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ticket-agent",
            "create-ticket",
            "--ticket-id",
            "CLI-0001",
            "--subject",
            "Cannot export invoice",
            "--body",
            "The export button fails every time I try it.",
            "--priority",
            "urgent",
            "--scripted-reply",
            "I retried it and the issue is still there.",
            "--tickets-file",
            str(tickets_path),
        ],
    )

    assert main() == 0

    payload = json.loads(tickets_path.read_text(encoding="utf-8"))
    assert payload[0]["ticket_id"] == "CLI-0001"
    assert payload[0]["priority"] == "urgent"
    assert payload[0]["pending_customer_replies"] == ["I retried it and the issue is still there."]

    stdout = capsys.readouterr().out
    assert "Created ticket CLI-0001" in stdout


def test_import_kb_command_writes_extra_file_and_is_loadable(tmp_path: Path, monkeypatch, capsys) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "tickets.json").write_text("[]", encoding="utf-8")
    (data_dir / "knowledge_base.json").write_text(
        json.dumps(
            [
                {
                    "article_id": "KB-100",
                    "title": "Base article",
                    "issue_type": "billing",
                    "information_type": "policy",
                    "keywords": ["invoice"],
                    "summary": "Base summary",
                    "resolution_steps": ["Collect the invoice ID."],
                    "customer_reply_template": "Please share the invoice ID."
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    source_path = tmp_path / "incoming_kb.json"
    source_path.write_text(
        json.dumps(
            [
                {
                    "article_id": "KB-900",
                    "title": "Imported article",
                    "issue_type": "integration",
                    "information_type": "product",
                    "keywords": ["oauth", "reconnect"],
                    "summary": "Imported summary",
                    "resolution_steps": ["Reconnect the integration."],
                    "customer_reply_template": "Please reconnect the integration."
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("TICKET_AGENT_ROOT", str(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ticket-agent",
            "import-kb",
            "--source",
            str(source_path),
        ],
    )

    assert main() == 0

    extra_path = data_dir / "knowledge_base_extra.json"
    payload = json.loads(extra_path.read_text(encoding="utf-8"))
    assert payload[0]["article_id"] == "KB-900"

    stdout = capsys.readouterr().out
    assert "loaded_total=2" in stdout


def test_repository_persists_runtime_mutations(tmp_path: Path) -> None:
    tickets_path = tmp_path / "tickets.json"
    tickets_path.write_text(
        json.dumps(
            [
                {
                    "ticket_id": "TICK-1001",
                    "customer_id": "CUST-001",
                    "subject": "Existing ticket",
                    "body": "Existing body",
                    "priority": "medium",
                    "status": "open",
                    "issue_type": None,
                    "resolution_summary": None,
                    "customer_message_history": [],
                    "pending_customer_replies": ["Still blocked."],
                    "response_history": [],
                    "tool_failures": {"search_kb": 1},
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    repository = TicketRepository(tickets_path)
    repository.append_response("TICK-1001", "Please try one more step.")
    repository.consume_next_customer_reply("TICK-1001")
    repository.set_issue_type("TICK-1001", "billing")
    repository.update_status("TICK-1001", "pending_customer", "Waiting for details")
    repository.consume_tool_failure("TICK-1001", "search_kb")

    reloaded = TicketRepository(tickets_path)
    persisted = reloaded.get("TICK-1001")
    assert persisted.response_history == ["Please try one more step."]
    assert persisted.customer_message_history == ["Still blocked."]
    assert persisted.issue_type == "billing"
    assert persisted.status == "pending_customer"
    assert persisted.resolution_summary == "Waiting for details"
    assert persisted.tool_failures["search_kb"] == 0


def test_import_kb_docs_command_converts_markdown_into_articles(tmp_path: Path, monkeypatch, capsys) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "tickets.json").write_text("[]", encoding="utf-8")
    (data_dir / "knowledge_base.json").write_text(
        json.dumps(
            [
                {
                    "article_id": "KB-100",
                    "title": "Base article",
                    "issue_type": "integration",
                    "information_type": "product",
                    "keywords": ["webhook"],
                    "summary": "Base summary",
                    "resolution_steps": ["Check webhook logs."],
                    "customer_reply_template": "Please check the webhook logs."
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "oauth_reconnect.md").write_text(
        "# OAuth reconnect guide\n\nReconnect the integration when consent expires.\n\n- Open the integration settings.\n- Reconnect the provider.\n- Retry the sync.\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("TICKET_AGENT_ROOT", str(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ticket-agent",
            "import-kb-docs",
            "--source",
            str(docs_dir),
            "--issue-type",
            "integration",
            "--information-type",
            "product",
        ],
    )

    assert main() == 0

    extra_path = data_dir / "knowledge_base_extra.json"
    payload = json.loads(extra_path.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["article_id"].startswith("KB-DOC-")
    assert payload[0]["title"] == "OAuth reconnect guide"
    assert payload[0]["resolution_steps"]

    stdout = capsys.readouterr().out
    assert "loaded_total=2" in stdout
