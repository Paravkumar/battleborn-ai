import json
from pathlib import Path

from ticket_agent.config import Settings
from ticket_agent.knowledge_base import KnowledgeBase


def test_search_prefers_grounded_issue_match_and_exposes_matched_terms() -> None:
    settings = Settings(use_ollama=False)
    kb = KnowledgeBase(settings.data_dir / "knowledge_base.json")

    hits = kb.search(
        query="Webhook delivery keeps timing out and signature validation fails",
        issue_type="integration",
        intent="troubleshoot_integration_failure",
        limit=3,
    )

    assert hits
    assert hits[0].article_id == "KB-120"
    assert "webhook" in hits[0].matched_terms
    assert hits[0].confidence_score >= 0.55


def test_search_matches_subscription_refund_requests() -> None:
    settings = Settings(use_ollama=False)
    kb = KnowledgeBase(settings.data_dir / "knowledge_base.json")

    hits = kb.search(
        query="I bought your subscription and I want my money back",
        issue_type="billing",
        intent="request_subscription_refund",
        limit=3,
    )

    assert hits
    assert hits[0].article_id == "KB-115"
    assert "subscription" in hits[0].matched_terms


def test_knowledge_base_loads_multiple_files_from_same_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    base_path = data_dir / "knowledge_base.json"
    extra_path = data_dir / "knowledge_base_extra.json"

    base_path.write_text(
        json.dumps(
            [
                {
                    "article_id": "KB-BASE",
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
    extra_path.write_text(
        json.dumps(
            [
                {
                    "article_id": "KB-EXTRA",
                    "title": "Extra article",
                    "issue_type": "integration",
                    "information_type": "product",
                    "keywords": ["webhook"],
                    "summary": "Extra summary",
                    "resolution_steps": ["Check webhook logs."],
                    "customer_reply_template": "Please check the webhook logs."
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    kb = KnowledgeBase(base_path)

    assert kb.article_count() == 2
    assert kb.has("KB-BASE")
    assert kb.has("KB-EXTRA")
