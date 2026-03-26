from __future__ import annotations

import json

from ticket_agent.few_shot import load_response_examples


def test_load_response_examples_extracts_grounded_resolved_examples(tmp_path) -> None:
    dataset_path = tmp_path / "teacher_runs.jsonl"
    example = {
        "messages": [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "ticket": {
                            "subject": "Webhook delivery keeps timing out",
                        }
                    }
                ),
            },
            {
                "role": "tool",
                "name": "classify_ticket",
                "content": json.dumps({"data": {"issue_type": "integration"}}),
            },
            {
                "role": "tool",
                "name": "draft_response",
                "content": json.dumps(
                    {
                        "data": {
                            "body": "Please verify the signing secret and review recent retry logs.",
                            "citations": ["KB-120"],
                            "needs_escalation": False,
                        }
                    }
                ),
            },
        ]
    }
    dataset_path.write_text(json.dumps(example) + "\n", encoding="utf-8")

    examples = load_response_examples(dataset_path, limit=2)

    assert len(examples) == 1
    assert examples[0]["issue_type"] == "integration"
    assert examples[0]["citations"] == ["KB-120"]
