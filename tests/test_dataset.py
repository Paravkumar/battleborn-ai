from pathlib import Path

from ticket_agent.agent import TicketResolutionAgent
from ticket_agent.config import Settings
from ticket_agent.dataset import execution_log_to_sft_example, sft_example_to_training_text
from ticket_agent.knowledge_base import KnowledgeBase
from ticket_agent.repository import TicketRepository
from ticket_agent.tools import TicketResolutionToolkit


FIXTURE_TICKETS = Path(__file__).resolve().parent / "fixtures" / "sample_tickets.json"


def build_agent() -> TicketResolutionAgent:
    settings = Settings(use_ollama=False)
    repository = TicketRepository(FIXTURE_TICKETS)
    kb = KnowledgeBase(settings.data_dir / "knowledge_base.json")
    toolkit = TicketResolutionToolkit(
        repository=repository,
        kb=kb,
        llm=None,
        student_model=settings.student_model,
        use_ollama=False,
        min_kb_confidence=settings.min_kb_confidence,
        min_classification_confidence=settings.min_classification_confidence,
    )
    return TicketResolutionAgent(settings=settings, repository=repository, toolkit=toolkit, llm=None)


def test_sft_export_contains_tool_schemas() -> None:
    agent = build_agent()
    log = agent.run("TICK-1001", "Resolve the customer ticket end-to-end.")
    example = execution_log_to_sft_example(log)
    assert example["tools"][0]["type"] == "function"
    assert example["tools"][0]["function"]["parameters"]["type"] == "object"
    first_tool_call = example["messages"][1]["tool_calls"][0]
    assert isinstance(first_tool_call["function"]["arguments"], dict)


def test_training_text_flattens_conversational_examples() -> None:
    agent = build_agent()
    log = agent.run("TICK-1001", "Resolve the customer ticket end-to-end.")
    example = execution_log_to_sft_example(log)

    training_text = sft_example_to_training_text(example)

    assert "Available tools:" in training_text
    assert "Assistant tool calls:" in training_text
    assert "Tool read_ticket:" in training_text
    assert "Assistant:" in training_text
