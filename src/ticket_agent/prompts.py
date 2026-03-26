from __future__ import annotations

import json

from ticket_agent.schemas import KnowledgeBaseArticle, Ticket


PLANNER_SYSTEM_PROMPT = """You are a workflow planner for customer ticket resolution.
Return JSON only.
Use only the provided tools.
Create a short ordered plan for resolving the ticket.
Always include these tools in the execution plan in this order:
read_ticket, classify_ticket, search_kb, draft_response, post_response, update_ticket_status.
Do not include escalate_ticket in the initial plan. Escalation is handled only if a step fails after retries.
The plan should prefer grounded resolution and keep escalation as a fallback if retries fail.
"""

RESPONSE_SYSTEM_PROMPT = """You are a customer support agent.
Draft a concise response grounded only in the provided ticket details and knowledge-base articles.
Do not invent steps that are not supported by the evidence.
Do not mention the words "knowledge base", "article", or raw article IDs in the customer-facing body.
Do not repeat background context the customer already saw in earlier turns.
Do not escalate on the first supported turn when the retrieved evidence gives the customer a clear next step.
Escalate after the supported steps have already been tried and the customer reports the issue is still unresolved, or when the request is clearly out of scope.
If the retrieved evidence is an intake or triage guide, ask only the targeted missing details needed for the next grounded step.
Only set needs_escalation to true when the evidence is completely irrelevant or when the customer explicitly says previous steps failed.
On the first turn, always try to help with the available KB articles before considering escalation.

CRITICAL: You MUST always populate the citations array with the article_ids of all KB articles you used to draft the response.
For example, if you used KB-100 and KB-101, set citations to ["KB-100", "KB-101"].
Never leave citations as an empty array when you have KB articles available.

Return JSON only with this exact structure:
{
  "body": "your customer-facing response text",
  "citations": ["KB-XXX", "KB-YYY"],
  "resolution_confidence": "high" or "medium" or "low",
  "needs_escalation": false,
  "next_action": "await_customer" or "resolve" or "escalate"
}
"""


def build_planner_user_prompt(goal: str, ticket: Ticket, tools: list[dict[str, object]]) -> str:
    return json.dumps(
        {
            "goal": goal,
            "ticket": {
                "ticket_id": ticket.ticket_id,
                "subject": ticket.subject,
                "body": ticket.body,
                "priority": ticket.priority,
            },
            "tools": tools,
            "instructions": [
                "Use between 5 and 6 steps.",
                "Start with reading and classifying the ticket.",
                "Include KB lookup and response drafting before posting a reply.",
                "Always end with update_ticket_status.",
                "Do not include escalate_ticket in the initial plan.",
                "Do not include tools outside the list.",
            ],
        },
        indent=2,
    )


def build_response_user_prompt(
    ticket: Ticket,
    articles: list[KnowledgeBaseArticle],
    current_customer_message: str,
    customer_messages: list[str] | None = None,
    response_history: list[str] | None = None,
    response_examples: list[dict[str, object]] | None = None,
) -> str:
    payload = {
        "ticket": {
            "ticket_id": ticket.ticket_id,
            "subject": ticket.subject,
            "body": ticket.body,
            "priority": ticket.priority,
            "issue_type": ticket.issue_type,
        },
        "conversation": {
            "current_customer_message": current_customer_message,
            "customer_messages": customer_messages or [ticket.body],
            "agent_responses": response_history or [],
        },
        "knowledge_base": [
            {
                "article_id": article.article_id,
                "title": article.title,
                "information_type": article.information_type,
                "summary": article.summary,
                "resolution_steps": article.resolution_steps,
            }
            for article in articles
        ],
        "instructions": [
            "Write a grounded support reply.",
            "Use the retrieved policy or product information only.",
            "Every customer-facing step must be directly supported by the retrieved summaries or resolution_steps.",
            "If enough evidence exists, provide concrete next steps for the customer.",
            "If the retrieved guidance is mainly for intake or triage, ask for the missing details directly instead of pretending the issue is already diagnosed.",
            "On the first turn, always try to help the customer with the available KB evidence - do not escalate unless the evidence is completely irrelevant.",
            "Only set needs_escalation to true after the customer reports that supported steps did not work, or when the request is clearly out of scope for the available KB.",
            "Set next_action to await_customer when the customer still needs to do a step or provide missing information.",
            "Set next_action to resolve only when the customer confirmed success or the issue is clearly resolved from the conversation.",
            "Set next_action to escalate only when escalation is truly necessary.",
            "Keep the response concise and professional.",
            "Do not mention the knowledge base, policy source, article labels, or article IDs in the visible reply body.",
            "On follow-up turns, answer only the new customer update instead of restating the earlier guidance.",
            "IMPORTANT: Always include the article_ids of ALL KB articles in the citations array. Never leave citations empty when KB articles are provided.",
            "Set resolution_confidence to 'high' or 'medium' when the KB evidence is relevant. Only use 'low' when the evidence does not match at all.",
        ],
    }
    if response_examples:
        payload["reference_examples"] = response_examples
    return json.dumps(payload, indent=2)
