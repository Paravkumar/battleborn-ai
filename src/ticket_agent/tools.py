from __future__ import annotations

import re
from typing import Any

from ticket_agent.knowledge_base import KnowledgeBase, tokenize
from ticket_agent.ollama_client import OllamaGateway
from ticket_agent.prompts import RESPONSE_SYSTEM_PROMPT, build_response_user_prompt
from ticket_agent.repository import TicketRepository
from ticket_agent.schemas import DraftedReply, KnowledgeBaseArticle, KnowledgeBaseHit, ToolOutcome


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_ticket",
            "description": "Load the current ticket details and customer context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string", "description": "The ticket identifier."}
                },
                "required": ["ticket_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_ticket",
            "description": "Classify the ticket into a supported issue type, infer user intent, and estimate confidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string", "description": "The ticket identifier."}
                },
                "required": ["ticket_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Search the approved knowledge base for a grounded resolution path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string", "description": "The ticket identifier."},
                    "query": {"type": "string", "description": "The KB search query."},
                    "issue_type": {
                        "type": ["string", "null"],
                        "description": "The classified issue type when available.",
                    },
                    "intent": {
                        "type": ["string", "null"],
                        "description": "The inferred user intent for the ticket.",
                    },
                    "classification_confidence": {
                        "type": ["number", "null"],
                        "description": "Confidence score from the intent classifier.",
                    },
                    "limit": {"type": "integer", "description": "Maximum number of articles to return."},
                },
                "required": ["ticket_id", "query", "issue_type", "intent", "classification_confidence", "limit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_response",
            "description": "Draft a customer-facing reply using the ticket and retrieved KB evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string", "description": "The ticket identifier."},
                    "article_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The KB article identifiers used as evidence.",
                    },
                },
                "required": ["ticket_id", "article_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "post_response",
            "description": "Post the drafted response back to the ticket.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string", "description": "The ticket identifier."},
                    "response_body": {"type": "string", "description": "The customer-facing reply text."},
                    "citations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "KB article identifiers cited in the reply.",
                    },
                },
                "required": ["ticket_id", "response_body", "citations"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_ticket_status",
            "description": "Update the ticket status after resolution or handoff.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string", "description": "The ticket identifier."},
                    "status": {"type": "string", "description": "The new ticket status."},
                    "resolution_summary": {
                        "type": "string",
                        "description": "A short summary of the resolution or handoff.",
                    },
                },
                "required": ["ticket_id", "status", "resolution_summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_ticket",
            "description": "Escalate the ticket with a clear explanation if grounded resolution fails.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string", "description": "The ticket identifier."},
                    "reason": {"type": "string", "description": "Why the ticket must be escalated."},
                },
                "required": ["ticket_id", "reason"],
            },
        },
    },
]


def _classify_issue(text: str) -> str:
    lowered = text.lower()
    # Product/device usage guidance should not be misrouted to billing just because of "ordered/bought".
    if any(keyword in lowered for keyword in ["how to use", "guide", "guidance", "manual", "setup", "set up", "connect", "pair", "pairing"]) and any(
        keyword in lowered
        for keyword in ["device", "headphone", "earphone", "earbud", "speaker", "laptop", "bluetooth", "usb", "adapter", "new product"]
    ):
        return "integration"
    if any(
        keyword in lowered
        for keyword in [
            # Password and account recovery
            "password",
            "reset password",
            "forgot password",
            "forgotten password",
            "change password",
            "locked",
            "unlock",
            "locked out",
            # Login and sign-in issues
            "login",
            "log in",
            "signin",
            "sign in",
            "sign-in",
            "can't login",
            "cant login",
            "cannot login",
            "can't log in",
            "cant log in",
            "cannot log in",
            "can't sign in",
            "cant sign in",
            "cannot sign in",
            "unable to login",
            "unable to log in",
            "unable to sign in",
            "not able to login",
            "not able to log in",
            "failed to login",
            "failed to log in",
            "login failed",
            "login issue",
            "login problem",
            "sign in issue",
            "sign in problem",
            # Account access issues
            "account access",
            "access my account",
            "access account",
            "my account",
            "can't access",
            "cant access",
            "cannot access",
            "no access",
            "lost access",
            "hacked",
            "compromised",
            "stolen account",
            "someone logged in",
            "suspicious activity",
            "security",
            # MFA and 2FA
            "mfa",
            "2fa",
            "two-factor",
            "two factor",
            "verification code",
            "authenticator",
            "magic link",
            "one-time code",
            "otp",
            # Invitations and verification
            "invite",
            "invitation",
            "verify email",
            "activation link",
            "verification link",
            "confirm email",
            # SSO and permissions
            "sso",
            "single sign on",
            "single sign-on",
            "saml",
            "oidc",
            "permission",
            "role",
            "access denied",
            "not authorized",
        ]
    ):
        return "access_management"
    if any(
        keyword in lowered
        for keyword in [
            # Invoice and billing
            "invoice",
            "billing",
            "bill",
            "billed",
            # Charges and payments
            "charge",
            "charged",
            "overcharged",
            "double charged",
            "charged twice",
            "payment",
            "pay",
            "paid",
            "paying",
            # Refunds
            "refund",
            "money back",
            "get my money",
            "return money",
            "want money",
            # Subscriptions
            "subscription",
            "subscribe",
            "subscribed",
            "cancel my plan",
            "cancel subscription",
            "cancel plan",
            "cancel account",
            "cancellation",
            "renewal",
            "renew",
            "auto-renew",
            "auto renew",
            # Pricing and plans
            "receipt",
            "vat",
            "tax id",
            "tax",
            "payment method",
            "card declined",
            "card",
            "credit card",
            "debit card",
            "trial",
            "free trial",
            "proration",
            "downgrade",
            "upgrade",
            "seat",
            "plan change",
            "change plan",
            "pricing",
            "price",
            "cost",
            # Product quality (often leads to refund requests)
            "worst",
            "terrible",
            "horrible",
            "bad quality",
            "poor quality",
            "defective",
            "broken product",
            "not as described",
            "want refund",
            "requesting refund",
            "request refund",
            # Purchases
            "buy",
            "bought",
            "purchase",
            "purchased",
            "order",
            "ordered",
        ]
    ):
        return "billing"
    if any(
        keyword in lowered
        for keyword in [
            "webhook",
            "api",
            "integration",
            "integrate",
            "sync",
            "syncing",
            "synchronize",
            "timeout",
            "timed out",
            "export",
            "exporting",
            "import",
            "importing",
            "csv",
            "excel",
            "spreadsheet",
            "oauth",
            "token",
            "api key",
            "api-key",
            "rate limit",
            "rate-limit",
            "429",
            "throttle",
            "throttling",
            "firewall",
            "allowlist",
            "allow list",
            "whitelist",
            "white list",
            "sftp",
            "ftp",
            "dns",
            "custom domain",
            "domain",
            "cname",
            "notification",
            "notifications",
            "delivery",
            "not receiving",
            "attachment",
            "upload",
            "uploading",
            "download",
            "downloading",
            "connection",
            "connect",
            "connecting",
            "endpoint",
            "callback",
            "third party",
            "third-party",
            "external",
        ]
    ):
        return "integration"
    return "unknown"


def _infer_intent(issue_type: str, text: str) -> tuple[str, float]:
    lowered = text.lower()
    if any(keyword in lowered for keyword in ["how to use", "guide", "guidance", "manual", "setup", "set up", "connect", "pair", "pairing"]) and any(
        keyword in lowered
        for keyword in ["device", "headphone", "earphone", "earbud", "speaker", "laptop", "bluetooth", "usb", "adapter", "new product"]
    ):
        return "product_usage_guidance", 0.92
    if any(phrase in lowered for phrase in ["dont want a refund", "don't want a refund", "do not want a refund", "no refund"]):
        if any(keyword in lowered for keyword in ["how to", "how do i", "use it", "setup", "set up", "integrate", "integration", "api"]):
            return "product_usage_guidance", 0.9
        return "billing_inquiry", 0.75
    if issue_type == "access_management":
        if any(keyword in lowered for keyword in ["mfa", "2fa", "two-factor", "two factor", "authenticator", "verification code", "otp", "one-time"]):
            return "recover_mfa_access", 0.9
        if any(keyword in lowered for keyword in ["sso", "single sign on", "single sign-on", "saml", "oidc"]):
            return "restore_sso_access", 0.88
        if any(keyword in lowered for keyword in ["permission", "role", "access denied", "not authorized", "unauthorized"]):
            return "request_role_access", 0.84
        if any(keyword in lowered for keyword in ["reset", "password", "unlock", "locked", "forgot", "forgotten", "change password"]):
            return "restore_account_access", 0.92
        if any(keyword in lowered for keyword in ["hacked", "compromised", "stolen", "suspicious", "security"]):
            return "account_security_issue", 0.88
        if any(keyword in lowered for keyword in ["login", "log in", "sign in", "signin", "can't access", "cant access", "cannot access", "unable to"]):
            return "restore_account_access", 0.85
        if any(keyword in lowered for keyword in ["invite", "invitation", "verify email", "activation"]):
            return "account_verification", 0.82
        return "account_access_question", 0.75
    if issue_type == "billing":
        if any(keyword in lowered for keyword in ["how to use", "guide", "guidance", "manual", "setup", "set up", "connect", "pair", "pairing"]):
            return "product_usage_guidance", 0.86
        # Check for subscription refund (combined subscription + refund keywords)
        if any(keyword in lowered for keyword in ["subscription", "subscribed"]) and any(keyword in lowered for keyword in ["money back", "refund", "want money", "cancel"]):
            return "request_subscription_refund", 0.88
        if any(keyword in lowered for keyword in ["money back", "refund", "want money", "get my money", "return money"]):
            return "request_subscription_refund", 0.9
        if any(keyword in lowered for keyword in ["charged twice", "duplicate", "double charged", "overcharged"]):
            return "resolve_duplicate_charge", 0.9
        if any(keyword in lowered for keyword in ["cancel", "cancellation", "stop subscription", "end subscription"]):
            return "cancel_subscription", 0.88
        if any(keyword in lowered for keyword in ["subscription", "renewal", "renew", "auto-renew"]):
            return "manage_subscription", 0.85
        if any(keyword in lowered for keyword in ["card declined", "payment method", "renewal failed", "payment failed", "can't pay", "cant pay"]):
            return "resolve_payment_failure", 0.87
        if any(keyword in lowered for keyword in ["receipt", "invoice copy", "vat", "tax id"]):
            return "billing_document_request", 0.82
        if any(keyword in lowered for keyword in ["upgrade", "downgrade", "proration", "seat", "plan", "change plan"]):
            return "manage_subscription_plan", 0.83
        if any(keyword in lowered for keyword in ["worst", "terrible", "horrible", "bad quality", "poor quality", "defective", "broken"]):
            return "product_complaint_refund", 0.85
        if any(keyword in lowered for keyword in ["bought", "purchase", "purchased", "order", "ordered", "buy"]):
            return "purchase_inquiry", 0.8
        if any(keyword in lowered for keyword in ["invoice", "charge", "charged", "payment", "paid", "bill", "billed"]):
            return "billing_inquiry", 0.8
        return "billing_inquiry", 0.75
    if issue_type == "integration":
        if any(keyword in lowered for keyword in ["api key", "token", "unauthorized", "401", "403", "authentication"]):
            return "restore_api_authentication", 0.89
        if any(keyword in lowered for keyword in ["rate limit", "429", "throttl", "too many requests"]):
            return "handle_api_rate_limit", 0.87
        if any(keyword in lowered for keyword in ["csv", "import", "column", "mapping", "validation", "excel", "spreadsheet"]):
            return "resolve_import_validation", 0.86
        if any(keyword in lowered for keyword in ["export", "download", "report", "stuck"]):
            return "resolve_export_delay", 0.84
        if any(keyword in lowered for keyword in ["oauth", "reconnect", "consent", "revoked"]):
            return "restore_oauth_connection", 0.86
        if any(keyword in lowered for keyword in ["sync", "duplicate record", "conflict", "syncing"]):
            return "troubleshoot_sync_conflict", 0.85
        if any(keyword in lowered for keyword in ["firewall", "allowlist", "allow list", "whitelist", "dns", "custom domain", "sftp", "cname"]):
            return "resolve_connectivity_block", 0.84
        if any(keyword in lowered for keyword in ["notification", "email delivery", "not receiving", "attachment", "upload", "uploading"]):
            return "troubleshoot_delivery_or_upload", 0.82
        if any(keyword in lowered for keyword in ["webhook", "api", "timeout", "timed out", "endpoint", "callback"]):
            return "troubleshoot_integration_failure", 0.88
        if any(keyword in lowered for keyword in ["connection", "connect", "connecting", "third party", "external"]):
            return "troubleshoot_connectivity", 0.8
        return "integration_question", 0.75
    # For unknown issue type, try to infer from text and give reasonable confidence
    # This prevents automatic escalation when we have some context
    if any(keyword in lowered for keyword in ["login", "log in", "sign in", "password", "account", "access"]):
        return "general_access_question", 0.6
    if any(keyword in lowered for keyword in ["refund", "money", "charge", "payment", "subscription", "cancel"]):
        return "general_billing_question", 0.6
    if any(keyword in lowered for keyword in ["api", "webhook", "integration", "sync", "export", "import"]):
        return "general_integration_question", 0.6
    return "general_inquiry", 0.5


CLARIFICATION_ARTICLES_BY_ISSUE = {
    "access_management": "KB-140",
    "billing": "KB-141",
    "integration": "KB-142",
}

TEAM_BY_ISSUE = {
    "access_management": "support specialist",
    "billing": "billing specialist",
    "integration": "technical support specialist",
}


def _looks_vague(text: str) -> bool:
    """Check if the request is too vague to process without clarification.
    
    Only returns True when there's genuinely not enough context to proceed.
    We want to be lenient and try to help customers when possible.
    """
    lowered = text.lower()
    tokens = tokenize(text)
    
    # Very short messages without actionable content
    if len(tokens) < 4:
        # But even short messages might have actionable keywords
        actionable_keywords = [
            "refund", "cancel", "login", "password", "reset", "charge",
            "subscription", "invoice", "receipt", "access", "locked",
            "payment", "upgrade", "downgrade", "api", "webhook", "export",
            "hacked", "compromised", "security", "mfa", "2fa", "sso",
            "account", "sign in", "signin", "log in", "verify", "verification",
        ]
        if any(kw in lowered for kw in actionable_keywords):
            return False
        return True
    
    # Only truly vague complaints without any actionable context
    # These are cases where we genuinely need more info
    purely_vague_patterns = [
        "it's bad",
        "this is bad",
        "help me",
        "need help",
        "something is wrong",
        "not happy",
        "very unhappy",
        "frustrated",
    ]
    
    # Check if the message is purely vague (just a complaint with no specifics)
    if any(pattern in lowered for pattern in purely_vague_patterns):
        # But if they also mention something specific, it's not vague
        specific_keywords = [
            "refund", "cancel", "login", "password", "account", "charge",
            "subscription", "payment", "invoice", "api", "export", "import",
            "email", "verification", "locked", "access", "mfa", "sso",
            "hacked", "compromised", "security", "sign in", "log in",
        ]
        if not any(kw in lowered for kw in specific_keywords):
            return True
    
    return False


def _infer_priority(text: str, issue_type: str) -> str:
    """Infer ticket priority based on content and urgency signals."""
    lowered = text.lower()
    
    # Urgent priority indicators
    urgent_keywords = [
        "urgent", "urgently", "emergency", "critical", "asap", "immediately",
        "right now", "can't wait", "cannot wait", "blocking", "blocked",
        "production down", "system down", "outage", "security breach",
        "hacked", "compromised", "stolen", "fraud", "unauthorized access",
        "data loss", "data breach", "legal", "compliance",
    ]
    if any(kw in lowered for kw in urgent_keywords):
        return "urgent"
    
    # High priority indicators
    high_keywords = [
        "high priority", "important", "need help today", "need this today",
        "deadline", "time sensitive", "as soon as possible", "quickly",
        "can't access", "cant access", "cannot access", "locked out",
        "can't login", "cant login", "cannot login", "can't log in",
        "payment failed", "charge failed", "double charged", "charged twice",
        "money back", "refund", "worst", "terrible", "horrible",
        "not working at all", "completely broken", "total failure",
    ]
    if any(kw in lowered for kw in high_keywords):
        return "high"
    
    # Low priority indicators
    low_keywords = [
        "when you have time", "no rush", "not urgent", "just wondering",
        "curious", "question about", "how do i", "how to", "information",
        "general question", "feedback", "suggestion", "feature request",
    ]
    if any(kw in lowered for kw in low_keywords):
        return "low"
    
    # Issue type based defaults
    if issue_type == "access_management":
        # Account access issues are typically high priority
        if any(kw in lowered for kw in ["login", "log in", "sign in", "password", "locked", "access"]):
            return "high"
    
    if issue_type == "billing":
        # Billing issues involving money are typically high
        if any(kw in lowered for kw in ["charge", "refund", "money", "payment"]):
            return "high"
    
    # Default to medium priority
    return "medium"


class TicketResolutionToolkit:
    def __init__(
        self,
        repository: TicketRepository,
        kb: KnowledgeBase,
        llm: OllamaGateway | None,
        student_model: str,
        use_ollama: bool,
        min_kb_confidence: float,
        min_classification_confidence: float,
        response_examples: list[dict[str, object]] | None = None,
    ) -> None:
        self.repository = repository
        self.kb = kb
        self.llm = llm
        self.student_model = student_model
        self.use_ollama = use_ollama
        self.min_kb_confidence = min_kb_confidence
        self.min_classification_confidence = min_classification_confidence
        self.response_examples = response_examples or []

    def available_tools(self) -> list[dict[str, Any]]:
        return TOOL_SPECS

    def execute(self, tool_name: str, **arguments: Any) -> ToolOutcome:
        handler = getattr(self, tool_name, None)
        if handler is None:
            return ToolOutcome(success=False, retryable=False, message=f"Unsupported tool: {tool_name}")
        return handler(**arguments)

    def _transient_failure(self, ticket_id: str, tool_name: str) -> ToolOutcome | None:
        if not self.repository.consume_tool_failure(ticket_id, tool_name):
            return None
        return ToolOutcome(
            success=False,
            retryable=True,
            message=f"Simulated transient failure in {tool_name}.",
        )

    def read_ticket(self, ticket_id: str) -> ToolOutcome:
        failure = self._transient_failure(ticket_id, "read_ticket")
        if failure:
            return failure
        ticket = self.repository.get(ticket_id)
        return ToolOutcome(
            success=True,
            message="Ticket loaded.",
            data={"ticket": ticket.model_dump(mode="json")},
        )

    def classify_ticket(self, ticket_id: str) -> ToolOutcome:
        failure = self._transient_failure(ticket_id, "classify_ticket")
        if failure:
            return failure
        ticket = self.repository.get(ticket_id)
        current_customer_message = self.repository.current_customer_message(ticket_id)
        ticket_text = f"{ticket.subject}\n{current_customer_message}"
        issue_type = _classify_issue(ticket_text)
        intent, confidence_score = _infer_intent(issue_type, ticket_text)
        priority = _infer_priority(ticket_text, issue_type)
        self.repository.set_issue_type(ticket_id, issue_type)
        self.repository.set_priority(ticket_id, priority)
        confidence = (
            "high"
            if confidence_score >= 0.85
            else "medium"
            if confidence_score >= self.min_classification_confidence
            else "low"
        )
        return ToolOutcome(
            success=True,
            message=f"Ticket classified as {issue_type} with {priority} priority.",
            data={
                "issue_type": issue_type,
                "intent": intent,
                "confidence": confidence,
                "confidence_score": confidence_score,
                "priority": priority,
                "in_scope": issue_type != "unknown",
            },
        )

    def search_kb(
        self,
        ticket_id: str,
        query: str,
        issue_type: str | None,
        intent: str | None,
        classification_confidence: float | None,
        limit: int = 3,
    ) -> ToolOutcome:
        failure = self._transient_failure(ticket_id, "search_kb")
        if failure:
            return failure
        hits = self.kb.search(query=query, issue_type=issue_type, intent=intent, limit=limit)
        if not hits:
            clarification = self._build_clarification_search_result(
                ticket_id=ticket_id,
                query=query,
                issue_type=issue_type,
                intent=intent,
                classification_confidence=classification_confidence,
            )
            if clarification is not None:
                return clarification
            return ToolOutcome(
                success=False,
                retryable=True,
                message="No grounded KB article matched the query.",
                data={"query": query, "issue_type": issue_type, "intent": intent},
            )
        top_hit = hits[0]
        if top_hit.confidence_score < self.min_kb_confidence:
            clarification = self._build_clarification_search_result(
                ticket_id=ticket_id,
                query=query,
                issue_type=issue_type,
                intent=intent,
                classification_confidence=classification_confidence,
            )
            if clarification is not None:
                return clarification
            return ToolOutcome(
                success=False,
                retryable=True,
                message="KB evidence is below the confidence threshold for automated resolution.",
                data={
                    "query": query,
                    "issue_type": issue_type,
                    "intent": intent,
                    "classification_confidence": classification_confidence,
                    "top_hit": top_hit.model_dump(mode="json"),
                },
            )
        return ToolOutcome(
            success=True,
            message=f"Retrieved {len(hits)} KB article(s).",
            data={
                "query": query,
                "issue_type": issue_type,
                "intent": intent,
                "classification_confidence": classification_confidence,
                "grounding_confidence": top_hit.confidence_score,
                "information_types": [hit.information_type for hit in hits],
                "hits": [hit.model_dump(mode="json") for hit in hits],
            },
        )

    def _build_clarification_search_result(
        self,
        ticket_id: str,
        query: str,
        issue_type: str | None,
        intent: str | None,
        classification_confidence: float | None,
    ) -> ToolOutcome | None:
        selection = self._select_clarification_hit(
            ticket_id=ticket_id,
            query=query,
            issue_type=issue_type,
            intent=intent,
            classification_confidence=classification_confidence,
        )
        if selection is None:
            return None
        hit, reason = selection
        return ToolOutcome(
            success=True,
            message="Retrieved clarification guidance.",
            data={
                "query": query,
                "issue_type": issue_type,
                "intent": intent,
                "classification_confidence": classification_confidence,
                "grounding_confidence": hit.confidence_score,
                "information_types": [hit.information_type],
                "needs_clarification": True,
                "clarification_reason": reason,
                "hits": [hit.model_dump(mode="json")],
            },
        )

    def _select_clarification_hit(
        self,
        ticket_id: str,
        query: str,
        issue_type: str | None,
        intent: str | None,
        classification_confidence: float | None,
    ) -> tuple[KnowledgeBaseHit, str] | None:
        ticket = self.repository.get(ticket_id)
        if ticket.response_history:
            return None

        article_id: str | None = None
        reason: str | None = None
        normalized_issue_type = issue_type or "unknown"
        if normalized_issue_type in CLARIFICATION_ARTICLES_BY_ISSUE:
            article_id = CLARIFICATION_ARTICLES_BY_ISSUE[normalized_issue_type]
            reason = f"The ticket appears to be {normalized_issue_type}, but more customer details are needed before a grounded resolution can be chosen."
        elif normalized_issue_type == "unknown" and _looks_vague(query):
            article_id = "KB-143"
            reason = "The request is still too broad for a grounded resolution, so the agent should collect the missing support details first."

        if not article_id or not self.kb.has(article_id):
            return None

        article = self.kb.get(article_id)
        article_text_tokens = tokenize(
            " ".join(
                [
                    article.title,
                    article.summary,
                    *article.keywords,
                    *article.resolution_steps,
                ]
            )
        )
        matched_terms = sorted(tokenize(query) & article_text_tokens)
        confidence = 0.62 if normalized_issue_type != "unknown" else 0.56
        score = max(4.5, float(len(matched_terms)) + 4.0)
        hit = KnowledgeBaseHit(
            article_id=article.article_id,
            title=article.title,
            issue_type=article.issue_type,
            information_type=article.information_type,
            summary=article.summary,
            matched_terms=matched_terms,
            score=score,
            confidence_score=confidence,
        )
        return hit, reason

    def draft_response(self, ticket_id: str, article_ids: list[str]) -> ToolOutcome:
        failure = self._transient_failure(ticket_id, "draft_response")
        if failure:
            return failure
        ticket = self.repository.get(ticket_id)
        articles = [self.kb.get(article_id) for article_id in article_ids if self.kb.has(article_id)]
        if not articles:
            return ToolOutcome(
                success=False,
                retryable=False,
                message="Cannot draft a grounded reply without KB evidence.",
            )
        reply = self._draft_with_model(ticket_id=ticket_id, articles=articles)
        reply = self._normalize_reply_action(ticket_id=ticket_id, reply=reply)
        reply = self._calibrate_reply(ticket_id=ticket_id, reply=reply, articles=articles)
        reply.body = self._sanitize_reply_body(reply.body)
        # Hard anti-loop guard: never send the exact same assistant message twice in a row.
        if ticket.response_history and ticket.response_history[-1].strip().lower() == reply.body.strip().lower():
            issue_type = ticket.issue_type or "general"
            follow_up_by_issue = {
                "integration": "Thanks for the update. To move this forward, share one concrete artifact: exact endpoint, request ID, or full error payload from the latest failed call.",
                "access_management": "Thanks for the update. To continue, share one concrete detail: exact login error text, account email, and whether 2FA prompt appears.",
                "billing": "Thanks for the update. To continue, share one concrete detail: order/subscription ID, transaction date, and the exact billing error shown.",
                "general": "Thanks for the update. To continue, share one concrete detail we do not have yet: exact error text/code, timestamp, and what changed after the previous step.",
            }
            reply.body = follow_up_by_issue.get(issue_type, follow_up_by_issue["general"])
            reply.needs_escalation = False
            reply.next_action = "await_customer"
            reply.resolution_confidence = "medium"
        # Auto-populate citations from provided articles if LLM omitted them
        # This fixes a common issue where the model generates good responses but forgets citations
        if not reply.citations and not reply.needs_escalation:
            reply.citations = [article.article_id for article in articles]
        validation_error = self._validate_draft_reply(reply, articles)
        if validation_error is not None:
            return ToolOutcome(
                success=False,
                retryable=False,
                message=validation_error,
                data={
                    **reply.model_dump(mode="json"),
                    "source": getattr(reply, "_source", "fallback"),
                },
            )
        if reply.needs_escalation or reply.resolution_confidence == "low":
            return ToolOutcome(
                success=False,
                retryable=False,
                message="Draft indicates escalation is required.",
                data={
                    **reply.model_dump(mode="json"),
                    "source": getattr(reply, "_source", "fallback"),
                },
            )
        return ToolOutcome(
            success=True,
            message="Grounded response drafted.",
            data={
                **reply.model_dump(mode="json"),
                "source": getattr(reply, "_source", "fallback"),
            },
        )

    def _draft_with_model(self, ticket_id: str, articles: list[KnowledgeBaseArticle]) -> DraftedReply:
        ticket = self.repository.get(ticket_id)
        current_customer_message = self.repository.current_customer_message(ticket_id)
        customer_messages = self.repository.customer_messages(ticket_id)
        if self.use_ollama and self.llm is not None and self.llm.is_ready():
            try:
                payload = self.llm.chat_json(
                    model=self.student_model,
                    messages=[
                        {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": build_response_user_prompt(
                                ticket,
                                articles,
                                current_customer_message=current_customer_message,
                                customer_messages=customer_messages,
                                response_history=ticket.response_history,
                                response_examples=self.response_examples,
                            ),
                        },
                    ],
                    schema=DraftedReply.model_json_schema(),
                )
                reply = DraftedReply.model_validate(payload)
                setattr(reply, "_source", "ollama")
                return reply
            except Exception:
                pass
        reply = self._fallback_draft_reply(
            ticket=ticket,
            current_customer_message=current_customer_message,
            articles=articles,
        )
        setattr(reply, "_source", "fallback")
        return reply

    def _fallback_draft_reply(
        self,
        ticket,
        current_customer_message: str,
        articles: list[KnowledgeBaseArticle],
    ) -> DraftedReply:
        primary = articles[0]
        lowered = current_customer_message.lower()
        citations = [article.article_id for article in articles]
        has_agent_history = bool(ticket.response_history)

        resolved_markers = [
            "it worked",
            "works now",
            "can sign in",
            "can log in",
            "can login",
            "able to log in",
            "able to login",
            "now able to log in",
            "now able to login",
            "logged in now",
            "issue is fixed",
            "resolved",
            "thanks, that fixed it",
            "refund received",
            "cancellation is confirmed",
            "received the invoice",
            "payment went through",
        ]
        unresolved_markers = [
            "still cannot",
            "still can't",
            "didn't work",
            "did not work",
            "still broken",
            "still failing",
            "still timing out",
        ]
        clarification_articles = {"KB-140", "KB-141", "KB-142", "KB-143"}

        if any(marker in lowered for marker in resolved_markers):
            return DraftedReply(
                body="Thanks for confirming the issue is resolved. I'm marking this ticket as resolved now. If the problem returns, reply here and we can reopen it.",
                citations=citations,
                resolution_confidence="high",
                needs_escalation=False,
                next_action="resolve",
            )

        if primary.issue_type == "billing" and has_agent_history:
            if primary.article_id == "KB-115":
                return DraftedReply(
                    body="Thanks for the update. I have enough context to hand this to the billing team for refund eligibility review and cancellation handling. I'm escalating it now so they can complete the refund decision.",
                    citations=citations,
                    resolution_confidence="medium",
                    needs_escalation=True,
                    next_action="escalate",
                )
            if any(marker in lowered for marker in unresolved_markers):
                return DraftedReply(
                    body="Thanks for the update. The supported billing steps did not fully resolve the issue, so I'm escalating this ticket to the billing team for manual review.",
                    citations=citations,
                    resolution_confidence="medium",
                    needs_escalation=True,
                    next_action="escalate",
                )
            if "pending" in lowered or "authorization" in lowered:
                return DraftedReply(
                    body="Thanks for confirming the second transaction is still a pending authorization. No correction is needed if only one charge settled. The pending authorization should clear automatically according to your bank's timeline.",
                    citations=citations,
                    resolution_confidence="high",
                    needs_escalation=False,
                    next_action="resolve",
                )
            if "both settled" in lowered or "two settled" in lowered or "both charges settled" in lowered:
                return DraftedReply(
                    body="Thanks for confirming both charges settled. This needs a billing correction and refund review, so I'm escalating this to the billing team for manual handling.",
                    citations=citations,
                    resolution_confidence="medium",
                    needs_escalation=True,
                    next_action="escalate",
                )

        if primary.issue_type == "integration" and has_agent_history and any(marker in lowered for marker in unresolved_markers):
            return DraftedReply(
                body="The webhook checks did not resolve the issue, so this needs escalation to a human engineer for deeper investigation.",
                citations=citations,
                resolution_confidence="low",
                needs_escalation=True,
                next_action="escalate",
            )
        if primary.issue_type == "integration" and has_agent_history:
            if any(marker in lowered for marker in ["401", "403", "unauthorized", "forbidden", "token", "key", "api key"]):
                return DraftedReply(
                    body="Thanks for confirming the auth error. Please verify token scope and environment match (sandbox vs production), then rotate the key once and retry. If the 401/403 persists after that, I'm escalating this to technical support with your latest details.",
                    citations=citations,
                    resolution_confidence="medium",
                    needs_escalation=True,
                    next_action="escalate",
                )
            return DraftedReply(
                body="Thanks for the update. I need one more detail to continue: share the exact API error response body or request ID so I can pinpoint the integration failure.",
                citations=citations,
                resolution_confidence="medium",
                needs_escalation=False,
                next_action="await_customer",
            )

        if primary.issue_type == "access_management" and has_agent_history and any(marker in lowered for marker in unresolved_markers):
            return DraftedReply(
                body="The account recovery steps did not restore access, so I'm escalating this ticket to a human support specialist for deeper account review.",
                citations=citations,
                resolution_confidence="low",
                needs_escalation=True,
                next_action="escalate",
            )

        if primary.article_id in clarification_articles:
            if has_agent_history:
                return DraftedReply(
                    body="Thanks for the additional details. I still do not have enough grounded evidence to resolve this automatically, so I'm escalating this to a human specialist for manual review.",
                    citations=citations,
                    resolution_confidence="low",
                    needs_escalation=True,
                    next_action="escalate",
                )
            return DraftedReply(
                body=primary.customer_reply_template,
                citations=citations,
                resolution_confidence="medium",
                needs_escalation=False,
                next_action="await_customer",
            )

        body = primary.customer_reply_template
        if ticket.priority in {"high", "urgent"}:
            body = f"Priority noted. {body}"
        if has_agent_history and ticket.response_history and ticket.response_history[-1].strip().lower() == body.strip().lower():
            return DraftedReply(
                body="Thanks for the update. I need one more specific detail to move forward: share the exact error message/code or confirm what changed after the previous step.",
                citations=citations,
                resolution_confidence="medium",
                needs_escalation=False,
                next_action="await_customer",
            )
        return DraftedReply(
            body=body,
            citations=citations,
            resolution_confidence="high" if primary.issue_type != "unknown" else "medium",
            needs_escalation=False,
            next_action="await_customer",
        )

    def _validate_draft_reply(self, reply: DraftedReply, articles: list[KnowledgeBaseArticle]) -> str | None:
        if not reply.body.strip():
            return "Drafted response body was empty."
        allowed_citations = {article.article_id for article in articles}
        invalid_citations = [citation for citation in reply.citations if citation not in allowed_citations]
        if invalid_citations:
            return f"Draft cited unsupported KB articles: {', '.join(invalid_citations)}."
        if not reply.citations and not reply.needs_escalation:
            return "Draft omitted KB citations."
        return None

    def _normalize_reply_action(self, ticket_id: str, reply: DraftedReply) -> DraftedReply:
        ticket = self.repository.get(ticket_id)
        current_customer_message = self.repository.current_customer_message(ticket_id).lower()
        body = reply.body.lower()

        if reply.needs_escalation:
            reply.next_action = "escalate"
            return reply

        if any(
            marker in current_customer_message
            for marker in [
                "can sign in",
                "can log in",
                "can login",
                "able to log in",
                "able to login",
                "now able to log in",
                "now able to login",
                "logged in now",
                "works now",
                "it worked",
                "resolved",
                "fixed",
            ]
        ):
            reply.next_action = "resolve"
            return reply

        if not ticket.response_history:
            await_customer_markers = [
                "please try",
                "please provide",
                "please share",
                "reply here",
                "confirm whether",
                "check whether",
                "let us know",
            ]
            if any(marker in body for marker in await_customer_markers):
                reply.next_action = "await_customer"
        return reply

    def _calibrate_reply(
        self,
        ticket_id: str,
        reply: DraftedReply,
        articles: list[KnowledgeBaseArticle],
    ) -> DraftedReply:
        ticket = self.repository.get(ticket_id)
        heuristic = self._fallback_draft_reply(
            ticket=ticket,
            current_customer_message=self.repository.current_customer_message(ticket_id),
            articles=articles,
        )

        if not ticket.response_history and (reply.needs_escalation or reply.resolution_confidence == "low"):
            setattr(heuristic, "_source", "fallback")
            return heuristic

        if ticket.response_history:
            if heuristic.next_action == "resolve" and reply.next_action != "resolve":
                setattr(heuristic, "_source", "fallback")
                return heuristic
            if heuristic.next_action == "escalate" and reply.next_action != "escalate":
                setattr(heuristic, "_source", "fallback")
                return heuristic

        return reply

    def _sanitize_reply_body(self, body: str) -> str:
        cleaned = body.strip()
        substitutions = [
            (
                r"(?i)\b(?:based on|according to|as per)\s+(?:our\s+)?knowledge[- ]base(?:\s+\([^)]+\))?[:,]?\s*",
                "",
            ),
            (
                r"(?i)\b(?:based on|according to|as per)\s+(?:the\s+)?(?:billing|account|access|product|policy)\s+(?:policy|guidance)[:,]?\s*",
                "",
            ),
            (r"\(\s*KB-[^)]+\)", ""),
            (r"(?i)\barticle\s+KB-\S+", ""),
        ]
        for pattern, replacement in substitutions:
            cleaned = re.sub(pattern, replacement, cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ ]{2,}", " ", cleaned)
        return cleaned.strip()

    def post_response(self, ticket_id: str, response_body: str, citations: list[str]) -> ToolOutcome:
        failure = self._transient_failure(ticket_id, "post_response")
        if failure:
            return failure
        body = response_body.strip()
        ticket = self.repository.append_response(ticket_id, body)
        return ToolOutcome(
            success=True,
            message="Response posted to ticket.",
            data={"response_history": list(ticket.response_history), "citations": list(citations)},
        )

    def update_ticket_status(self, ticket_id: str, status: str, resolution_summary: str) -> ToolOutcome:
        failure = self._transient_failure(ticket_id, "update_ticket_status")
        if failure:
            return failure
        ticket = self.repository.update_status(ticket_id, status=status, resolution_summary=resolution_summary)
        return ToolOutcome(
            success=True,
            message=f"Ticket status updated to {status}.",
            data={"status": ticket.status, "resolution_summary": ticket.resolution_summary},
        )

    def escalate_ticket(self, ticket_id: str, reason: str) -> ToolOutcome:
        failure = self._transient_failure(ticket_id, "escalate_ticket")
        if failure:
            return failure
        customer_message = self._build_customer_escalation_message(ticket_id=ticket_id, reason=reason)
        ticket = self.repository.update_status(ticket_id, status="escalated", resolution_summary=reason)
        self.repository.append_response(ticket_id, customer_message)
        return ToolOutcome(
            success=True,
            message="Ticket escalated to human support.",
            data={"status": ticket.status, "reason": reason, "customer_message": customer_message},
        )

    def _build_customer_escalation_message(self, ticket_id: str, reason: str) -> str:
        ticket = self.repository.get(ticket_id)
        team = TEAM_BY_ISSUE.get(ticket.issue_type or "", "human support specialist")
        if ticket.response_history:
            return (
                f"Thanks for the additional details. I'm escalating this to a {team} now and "
                "including the conversation history and the steps already attempted so they can continue from here."
            )
        if "manual review" in reason.lower() or ticket.issue_type == "billing":
            return (
                f"I'm escalating this to a {team} now for manual review. "
                "They'll continue from the details you already shared."
            )
        return (
            f"I'm escalating this to a {team} now because this needs hands-on review beyond the supported automated steps. "
            "They'll continue from the details already captured in this ticket."
        )
