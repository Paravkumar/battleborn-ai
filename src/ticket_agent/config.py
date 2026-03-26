from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


def _discover_root_dir() -> Path:
    env_root = os.getenv("TICKET_AGENT_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "data" / "tickets.json").exists():
            return candidate

    cwd = Path.cwd().resolve()
    if (cwd / "data" / "tickets.json").exists():
        return cwd

    package_guess = Path(__file__).resolve().parents[2]
    if (package_guess / "data" / "tickets.json").exists():
        return package_guess

    return cwd


ROOT_DIR = _discover_root_dir()


class Settings(BaseModel):
    data_dir: Path = Field(default_factory=lambda: _discover_root_dir() / "data")
    output_dir: Path = Field(default_factory=lambda: _discover_root_dir() / "outputs")
    teacher_examples_path: Path = Field(default_factory=lambda: _discover_root_dir() / "artifacts" / "teacher_runs.jsonl")
    ollama_host: str = Field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    student_model: str = Field(default_factory=lambda: os.getenv("OLLAMA_STUDENT_MODEL", "llama3.1:70b"))
    teacher_model: str = Field(default_factory=lambda: os.getenv("OLLAMA_TEACHER_MODEL", "llama3.1:70b"))
    max_response_examples: int = Field(default_factory=lambda: int(os.getenv("MAX_RESPONSE_EXAMPLES", "2")))
    max_conversation_turns: int = Field(default_factory=lambda: int(os.getenv("MAX_CONVERSATION_TURNS", "4")))
    max_retries_per_step: int = 2
    search_results_limit: int = 3
    min_kb_confidence: float = Field(default_factory=lambda: float(os.getenv("MIN_KB_CONFIDENCE", "0.55")))
    min_classification_confidence: float = Field(default_factory=lambda: float(os.getenv("MIN_CLASSIFICATION_CONFIDENCE", "0.45")))
    use_ollama: bool = True

    def ensure_output_dir(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir
