from __future__ import annotations

import random
from dataclasses import dataclass

from ticket_agent.schemas import Ticket


PRODUCTS = ["Nimbus Desk", "Atlas Console", "Harbor Sync", "Vector Billing"]
ACCESS_SUBJECTS = [
    "Locked out after repeated password attempts",
    "Need help to unlock my login",
    "Password reset link not helping me sign in",
    "Cannot sign in after password mismatch alerts",
]
ACCESS_BODIES = [
    "I entered the wrong password several times and now my login is locked. Please help me unlock the account for {product} today.",
    "My team lead asked me to reset the password, but I am still unable to sign in to {product}. I need access before the next customer call.",
    "The password reset email arrived, but after using it I still cannot log in to {product}. Can you help me unlock the profile?",
    "I think the sign in flow locked me out after too many password attempts on {product}. Please help me regain access quickly.",
]

BILLING_SUBJECTS = [
    "I was charged twice for the same invoice",
    "Duplicate billing charge on invoice {invoice_id}",
    "Need help with a refund for double charge",
    "Question about duplicate invoice payment",
]
BILLING_BODIES = [
    "I paid invoice {invoice_id} for {product}, and now I see two charges on my card. Please confirm whether one is a duplicate or a pending authorization.",
    "Our finance team noticed a double charge tied to invoice {invoice_id}. We need the billing policy and refund timeline for {product}.",
    "I was billed twice for invoice {invoice_id} yesterday. Please explain the duplicate charge and what refund steps apply.",
    "There are two settled charges for invoice {invoice_id} on my account. I need a billing correction and refund guidance.",
]

INTEGRATION_SUBJECTS = [
    "Webhook delivery keeps timing out",
    "API integration fails webhook signature validation",
    "Need help with webhook retry failures",
    "Our integration endpoint is timing out",
]
INTEGRATION_BODIES = [
    "The webhook delivery for {product} started timing out against {endpoint}. Please help us validate the signing secret and timeout settings.",
    "Our API integration to {product} is failing with webhook signature errors. We need the product troubleshooting steps.",
    "Recent webhook deliveries to {endpoint} keep failing after retries. Please share the supported checks for timeout and signature mismatches.",
    "We use {product} webhooks and the delivery logs show repeated timeout failures. Please help us troubleshoot the integration.",
]

ESCALATION_SUBJECTS = [
    "Custom warehouse routing engine broke after midnight",
    "Need root cause for private ERP batch mapping issue",
    "Internal procurement workflow creates invalid serial mappings",
    "Custom SFTP reconciliation job is corrupting downstream records",
]
ESCALATION_BODIES = [
    "Our private warehouse routing engine started producing invalid serial mappings after the midnight batch. This is a custom workflow with no standard runbook and needs specialist review.",
    "A custom ERP batch job now creates mismatched allocation records after midnight. We need root cause analysis, and I do not think this matches any standard product playbook.",
    "Our internal procurement workflow is generating broken serial assignments across systems. This process is heavily customized and needs a human engineer.",
    "A private SFTP reconciliation job is corrupting downstream records in a way we have never seen before. Please escalate this beyond first-line support.",
]

ENDPOINTS = [
    "https://hooks.customer-a.example/webhook",
    "https://sync.partner-b.example/events",
    "https://api.vendor-c.example/callbacks",
    "https://services.example-d.io/webhooks/support",
]

ACCESS_SUCCESS_REPLIES = [
    "I used the reset link and I can sign in again now. Thanks.",
    "That worked. I can sign in now and the account is back to normal.",
]
ACCESS_FAILURE_REPLIES = [
    "I followed the reset steps but I still cannot sign in to the account.",
    "I tried the reset flow and it still did not work for me.",
]

BILLING_PENDING_REPLIES = [
    "I checked the card activity. Only one charge settled and the second one is still a pending authorization.",
    "The second transaction still shows as a pending authorization and only one charge settled.",
]
BILLING_SETTLED_REPLIES = [
    "I checked again and both charges settled on the card statement.",
    "Both charges settled and I still need the billing correction.",
]

INTEGRATION_SUCCESS_REPLIES = [
    "We rotated the signing secret and retried the delivery. It works now.",
    "After updating the endpoint settings, the webhook works now.",
]
INTEGRATION_FAILURE_REPLIES = [
    "We checked the signing secret and timeout settings, but deliveries are still failing.",
    "We tried those checks and the webhook is still timing out.",
]


@dataclass(frozen=True)
class SyntheticCounts:
    access: int
    billing: int
    integration: int
    escalation: int

    @property
    def total(self) -> int:
        return self.access + self.billing + self.integration + self.escalation


def generate_synthetic_tickets(counts: SyntheticCounts, seed: int = 7) -> list[Ticket]:
    if counts.total <= 0:
        raise ValueError("At least one synthetic ticket must be requested.")

    rng = random.Random(seed)
    tickets: list[Ticket] = []
    sequence = 1

    for index in range(counts.access):
        tickets.append(_build_access_ticket(sequence, index, rng))
        sequence += 1
    for index in range(counts.billing):
        tickets.append(_build_billing_ticket(sequence, index, rng))
        sequence += 1
    for index in range(counts.integration):
        tickets.append(_build_integration_ticket(sequence, index, rng))
        sequence += 1
    for index in range(counts.escalation):
        tickets.append(_build_escalation_ticket(sequence, index, rng))
        sequence += 1

    rng.shuffle(tickets)
    return tickets


def _ticket_base(
    sequence: int,
    subject: str,
    body: str,
    priority: str,
    tool_failures: dict[str, int],
    pending_customer_replies: list[str] | None = None,
) -> Ticket:
    ticket_number = 2000 + sequence
    return Ticket(
        ticket_id=f"TICK-{ticket_number:04d}",
        customer_id=f"CUST-SYN-{ticket_number:04d}",
        subject=subject,
        body=body,
        priority=priority,
        status="open",
        issue_type=None,
        resolution_summary=None,
        pending_customer_replies=pending_customer_replies or [],
        response_history=[],
        tool_failures=tool_failures,
    )


def _build_access_ticket(sequence: int, index: int, rng: random.Random) -> Ticket:
    product = rng.choice(PRODUCTS)
    subject = rng.choice(ACCESS_SUBJECTS)
    body = rng.choice(ACCESS_BODIES).format(product=product)
    failures = {"search_kb": 1} if index % 4 == 0 else {}
    priority = rng.choice(["medium", "high", "high"])
    follow_up = rng.choice(ACCESS_FAILURE_REPLIES if index % 5 == 0 else ACCESS_SUCCESS_REPLIES)
    return _ticket_base(sequence, subject, body, priority, failures, pending_customer_replies=[follow_up])


def _build_billing_ticket(sequence: int, index: int, rng: random.Random) -> Ticket:
    product = rng.choice(PRODUCTS)
    invoice_id = f"INV-{rng.randint(1200, 9999)}"
    subject = rng.choice(BILLING_SUBJECTS).format(invoice_id=invoice_id)
    body = rng.choice(BILLING_BODIES).format(product=product, invoice_id=invoice_id)
    failures = {"search_kb": 1} if index % 3 == 0 else {}
    if index % 5 == 0:
        failures["post_response"] = 1
    priority = rng.choice(["medium", "high", "high"])
    follow_up = rng.choice(BILLING_SETTLED_REPLIES if index % 4 == 0 else BILLING_PENDING_REPLIES)
    return _ticket_base(sequence, subject, body, priority, failures, pending_customer_replies=[follow_up])


def _build_integration_ticket(sequence: int, index: int, rng: random.Random) -> Ticket:
    product = rng.choice(PRODUCTS)
    endpoint = rng.choice(ENDPOINTS)
    subject = rng.choice(INTEGRATION_SUBJECTS)
    body = rng.choice(INTEGRATION_BODIES).format(product=product, endpoint=endpoint)
    failures = {"search_kb": 1} if index % 4 == 0 else {}
    if index % 6 == 0:
        failures["draft_response"] = 1
    priority = rng.choice(["medium", "high", "urgent"])
    follow_up = rng.choice(INTEGRATION_FAILURE_REPLIES if index % 4 == 0 else INTEGRATION_SUCCESS_REPLIES)
    return _ticket_base(sequence, subject, body, priority, failures, pending_customer_replies=[follow_up])


def _build_escalation_ticket(sequence: int, index: int, rng: random.Random) -> Ticket:
    subject = rng.choice(ESCALATION_SUBJECTS)
    body = rng.choice(ESCALATION_BODIES)
    failures = {"search_kb": 1} if index % 6 == 0 else {}
    priority = rng.choice(["high", "urgent", "urgent"])
    return _ticket_base(sequence, subject, body, priority, failures)
