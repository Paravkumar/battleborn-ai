from __future__ import annotations

import json
import re
from pathlib import Path

from ticket_agent.schemas import KnowledgeBaseArticle, KnowledgeBaseHit


STOPWORDS = {
    "a",
    "an",
    "and",
    "after",
    "am",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "the",
    "to",
    "today",
    "there",
    "was",
    "we",
}

CLARIFICATION_ARTICLE_IDS = {"KB-140", "KB-141", "KB-142", "KB-143"}


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS
    }


def keyword_in_query(keyword: str, normalized_query: str) -> bool:
    pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
    return re.search(pattern, normalized_query) is not None


class KnowledgeBase:
    def __init__(self, kb_path: Path) -> None:
        self.source_paths = self._resolve_source_paths(kb_path)
        self._articles = self._load_articles(self.source_paths)
        self._by_id = {article.article_id: article for article in self._articles}

    @staticmethod
    def _resolve_source_paths(kb_path: Path) -> list[Path]:
        if kb_path.is_dir():
            return sorted(path for path in kb_path.glob("knowledge_base*.json") if path.is_file())

        if kb_path.name == "knowledge_base.json" and kb_path.parent.exists():
            sibling_paths = sorted(
                path
                for path in kb_path.parent.glob("knowledge_base*.json")
                if path.is_file()
            )
            if sibling_paths:
                return sibling_paths

        return [kb_path]

    @staticmethod
    def _load_articles(paths: list[Path]) -> list[KnowledgeBaseArticle]:
        articles: list[KnowledgeBaseArticle] = []
        seen_ids: set[str] = set()
        for path in paths:
            if not path.exists():
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError(f"KB file must contain a JSON list: {path}")
            for item in raw:
                article = KnowledgeBaseArticle.model_validate(item)
                if article.article_id in seen_ids:
                    raise ValueError(f"Duplicate KB article_id detected: {article.article_id}")
                seen_ids.add(article.article_id)
                articles.append(article)
        return articles

    def get(self, article_id: str) -> KnowledgeBaseArticle:
        return self._by_id[article_id]

    def has(self, article_id: str) -> bool:
        return article_id in self._by_id

    def article_count(self) -> int:
        return len(self._articles)

    def articles(self) -> list[KnowledgeBaseArticle]:
        return list(self._articles)

    def search(self, query: str, issue_type: str | None = None, intent: str | None = None, limit: int = 3) -> list[KnowledgeBaseHit]:
        query_tokens = tokenize(query)
        intent_tokens = tokenize((intent or "").replace("_", " "))
        normalized_query = query.lower()
        ranked: list[KnowledgeBaseHit] = []
        for article in self._articles:
            if article.article_id in CLARIFICATION_ARTICLE_IDS:
                continue
            title_tokens = tokenize(article.title)
            summary_tokens = tokenize(article.summary)
            keyword_tokens = tokenize(" ".join(article.keywords))
            step_tokens = tokenize(" ".join(article.resolution_steps))
            article_tokens = title_tokens | summary_tokens | keyword_tokens | step_tokens

            title_overlap = query_tokens & title_tokens
            summary_overlap = query_tokens & summary_tokens
            keyword_overlap = query_tokens & keyword_tokens
            step_overlap = query_tokens & step_tokens
            matched_terms = sorted(query_tokens & article_tokens)
            overlap = len(matched_terms)
            exact_issue_match = bool(issue_type and issue_type != "unknown" and article.issue_type == issue_type)
            issue_bonus = 2.5 if exact_issue_match else 0.0
            phrase_bonus = min(2.0, sum(1.0 for keyword in article.keywords if keyword_in_query(keyword, normalized_query)))
            intent_bonus = 1.0 if intent_tokens and intent_tokens & article_tokens else 0.0
            weighted_overlap = (
                (len(title_overlap) * 3.0)
                + (len(keyword_overlap) * 2.5)
                + (len(summary_overlap) * 1.5)
                + (len(step_overlap) * 1.0)
            )
            score = float(weighted_overlap + issue_bonus + phrase_bonus + intent_bonus)
            if not self._is_grounded_match(
                overlap=overlap,
                weighted_overlap=weighted_overlap,
                issue_type=issue_type,
                article_issue_type=article.issue_type,
            ):
                continue
            confidence_score = min(1.0, (weighted_overlap / 8.0) + (0.2 if exact_issue_match else 0.0) + (0.1 if phrase_bonus else 0.0))
            ranked.append(
                KnowledgeBaseHit(
                    article_id=article.article_id,
                    title=article.title,
                    issue_type=article.issue_type,
                    information_type=article.information_type,
                    summary=article.summary,
                    matched_terms=matched_terms,
                    score=score,
                    confidence_score=confidence_score,
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]

    def _is_grounded_match(self, overlap: int, weighted_overlap: float, issue_type: str | None, article_issue_type: str) -> bool:
        if overlap <= 0:
            return False
        if issue_type and issue_type != "unknown":
            return article_issue_type == issue_type and weighted_overlap >= 2.5
        return overlap >= 2 and weighted_overlap >= 4.0
