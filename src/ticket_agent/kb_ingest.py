from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

from ticket_agent.knowledge_base import STOPWORDS
from ticket_agent.schemas import KnowledgeBaseArticle


SUPPORTED_DOC_EXTENSIONS = {".md", ".markdown", ".txt", ".rst"}


def discover_doc_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.exists():
        raise FileNotFoundError(source)
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_DOC_EXTENSIONS
    )


def load_kb_articles_from_docs(
    source: Path,
    issue_type: str,
    information_type: str = "hybrid",
    article_prefix: str = "KB-DOC",
) -> list[KnowledgeBaseArticle]:
    doc_paths = discover_doc_paths(source)
    if not doc_paths:
        raise ValueError(f"No supported documentation files were found in {source}.")
    return [
        article_from_doc(
            path=path,
            issue_type=issue_type,
            information_type=information_type,
            article_prefix=article_prefix,
        )
        for path in doc_paths
    ]


def article_from_doc(
    path: Path,
    issue_type: str,
    information_type: str = "hybrid",
    article_prefix: str = "KB-DOC",
) -> KnowledgeBaseArticle:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    title = _extract_title(lines, path)
    summary = _extract_summary(text, title)
    steps = _extract_resolution_steps(lines, summary)
    keywords = _extract_keywords(title=title, summary=summary, steps=steps)
    reply_template = _build_customer_reply_template(title=title, steps=steps)
    return KnowledgeBaseArticle(
        article_id=_article_id_from_path(path=path, prefix=article_prefix),
        title=title,
        issue_type=issue_type,
        information_type=information_type,
        keywords=keywords,
        summary=summary,
        resolution_steps=steps,
        customer_reply_template=reply_template,
    )


def _article_id_from_path(path: Path, prefix: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", path.stem).strip("-").upper()[:24] or "DOC"
    suffix = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:4].upper()
    return f"{prefix}-{slug}-{suffix}"


def _extract_title(lines: list[str], path: Path) -> str:
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()[:120]
    for line in lines:
        stripped = line.strip()
        if stripped:
            return stripped[:120]
    return path.stem.replace("_", " ").replace("-", " ").title()[:120]


def _extract_summary(text: str, title: str) -> str:
    normalized = text.replace("\r\n", "\n")
    paragraphs = [re.sub(r"\s+", " ", chunk.strip()) for chunk in re.split(r"\n\s*\n", normalized) if chunk.strip()]
    for paragraph in paragraphs:
        candidate = paragraph.lstrip("#").strip()
        if not candidate or candidate == title:
            continue
        if re.match(r"^[-*]\s+", candidate):
            continue
        if re.match(r"^\d+[.)]\s+", candidate):
            continue
        return candidate[:320]
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", normalized).strip())
    fallback = " ".join(sentence for sentence in sentences[:2] if sentence).strip()
    return (fallback or title)[:320]


def _extract_resolution_steps(lines: list[str], summary: str) -> list[str]:
    steps: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        bullet = re.match(r"^[-*+]\s+(.*)$", stripped)
        ordered = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        content = bullet.group(1).strip() if bullet else ordered.group(1).strip() if ordered else None
        if content:
            steps.append(content.rstrip(".") + ".")
    if steps:
        return steps[:5]

    summary_sentences = {sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", summary) if sentence.strip()}
    prose = " ".join(line.strip() for line in lines if line.strip() and not line.strip().startswith("#"))
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", prose) if sentence.strip()]
    generated = [sentence if sentence.endswith((".", "!", "?")) else sentence + "." for sentence in sentences if sentence not in summary_sentences]
    return generated[:4] or ["Review the imported guidance and apply the documented next step."]


def _extract_keywords(title: str, summary: str, steps: list[str]) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", " ".join([title, summary, *steps]).lower())
    counts = Counter(token for token in tokens if token not in STOPWORDS and len(token) > 2)
    ranked = [token for token, _ in counts.most_common(8)]
    return ranked or ["support", "troubleshooting"]


def _build_customer_reply_template(title: str, steps: list[str]) -> str:
    if not steps:
        return f"Please review the {title.lower()} guidance and let me know what you observe next."
    leading_steps = " ".join(steps[:2]).strip()
    return f"Please try the following next: {leading_steps} Then let me know what happened."
