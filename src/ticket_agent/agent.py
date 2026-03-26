from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ticket_agent.config import Settings
from ticket_agent.ollama_client import OllamaGateway
from ticket_agent.prompts import PLANNER_SYSTEM_PROMPT, build_planner_user_prompt
from ticket_agent.repository import TicketRepository
from ticket_agent.schemas import ExecutionLog, ExecutionStepRecord, Plan, PlanStep, ToolOutcome
from ticket_agent.tools import TicketResolutionToolkit


class TicketResolutionAgent:
    REQUIRED_PLAN_TOOLS = [
        "read_ticket",
        "classify_ticket",
        "search_kb",
        "draft_response",
        "post_response",
        "update_ticket_status",
    ]

    DEFAULT_PURPOSES = {
        "read_ticket": "Load the ticket contents.",
        "classify_ticket": "Identify the ticket issue type.",
        "search_kb": "Retrieve grounded knowledge-base guidance.",
        "draft_response": "Draft a support reply using KB evidence.",
        "post_response": "Post the drafted response to the ticket.",
        "update_ticket_status": "Mark the ticket as resolved.",
    }

    def __init__(
        self,
        settings: Settings,
        repository: TicketRepository,
        toolkit: TicketResolutionToolkit,
        llm: OllamaGateway | None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.toolkit = toolkit
        self.llm = llm

    def run(self, ticket_id: str, goal: str) -> ExecutionLog:
        ticket = self.repository.get(ticket_id)
        plan, plan_source = self._build_plan(goal=goal, ticket_id=ticket_id)
        log = ExecutionLog(
            goal=goal,
            plan_source=plan_source,
            ticket_id=ticket_id,
            ticket_snapshot=ticket.model_dump(mode="json"),
            model_used=self.settings.student_model if self.settings.use_ollama else None,
            started_at=datetime.now(timezone.utc),
            plan=plan,
        )

        state: dict[str, object] = {"ticket_id": ticket_id}
        for conversation_turn in range(1, self.settings.max_conversation_turns + 1):
            state["conversation_turn"] = conversation_turn
            for step in plan.steps:
                result = self._run_step(
                    step=step,
                    state=state,
                    ticket_id=ticket_id,
                    log=log,
                    conversation_turn=conversation_turn,
                )
                if result is None:
                    continue
                if result.success:
                    self._apply_step_result(step.tool, result, state)
                    continue
                if not self._escalate_after_failure(
                    step=step,
                    failure=result,
                    state=state,
                    ticket_id=ticket_id,
                    log=log,
                    conversation_turn=conversation_turn,
                ):
                    log.status = "failed"
                    log.final_ticket_status = self.repository.get(ticket_id).status
                    log.response_source = self._extract_response_source(state)
                    log.finished_at = datetime.now(timezone.utc)
                    return log
                log.status = "escalated"
                log.final_ticket_status = self.repository.get(ticket_id).status
                log.final_response = state.get("escalation_reason")
                log.response_source = self._extract_response_source(state)
                log.finished_at = datetime.now(timezone.utc)
                return log

            ticket = self.repository.get(ticket_id)
            if ticket.status == "resolved":
                log.status = "completed"
                log.final_ticket_status = ticket.status
                log.final_response = self._extract_final_response(state)
                log.response_source = self._extract_response_source(state)
                log.finished_at = datetime.now(timezone.utc)
                return log

            if ticket.status == "pending_customer":
                if self._consume_customer_reply(ticket_id=ticket_id, log=log, conversation_turn=conversation_turn):
                    self._reset_turn_state(state)
                    continue
                log.status = "waiting_on_customer"
                log.final_ticket_status = ticket.status
                log.final_response = self._extract_final_response(state)
                log.response_source = self._extract_response_source(state)
                log.finished_at = datetime.now(timezone.utc)
                return log

        if self._escalate_due_to_turn_limit(ticket_id=ticket_id, state=state, log=log):
            log.status = "escalated"
        else:
            log.status = "failed"
        log.final_ticket_status = self.repository.get(ticket_id).status
        log.final_response = self._extract_final_response(state)
        log.response_source = self._extract_response_source(state)
        log.finished_at = datetime.now(timezone.utc)
        return log

    def save_log(self, log: ExecutionLog) -> Path:
        output_dir = self.settings.ensure_output_dir()
        timestamp = log.started_at.strftime("%Y%m%dT%H%M%S%fZ")
        path = output_dir / f"{log.ticket_id}_{timestamp}.json"
        path.write_text(json.dumps(log.model_dump(mode="json"), indent=2), encoding="utf-8")
        return path

    def _build_plan(self, goal: str, ticket_id: str) -> tuple[Plan, str]:
        ticket = self.repository.get(ticket_id)
        fallback = self._default_plan()
        if not self.settings.use_ollama or self.llm is None or not self.llm.is_ready():
            return fallback, "fallback"
        try:
            payload = self.llm.chat_json(
                model=self.settings.student_model,
                messages=[
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_planner_user_prompt(goal, ticket, self.toolkit.available_tools()),
                    },
                ],
                schema=Plan.model_json_schema(),
            )
            plan = Plan.model_validate(payload)
            if self._plan_is_usable(plan):
                return self._normalize_plan(plan), "ollama"
        except Exception:
            pass
        return fallback, "fallback"

    def _plan_is_usable(self, plan: Plan) -> bool:
        valid_tools = {item["function"]["name"] for item in self.toolkit.available_tools()}
        planned_tools = [step.tool for step in plan.steps]
        if not all(tool in valid_tools for tool in planned_tools):
            return False
        core_tools = self.REQUIRED_PLAN_TOOLS[:-1]
        if not all(tool in planned_tools for tool in core_tools):
            return False
        present_order = [tool for tool in planned_tools if tool in self.REQUIRED_PLAN_TOOLS]
        required_indexes = [present_order.index(tool) for tool in core_tools]
        return required_indexes == sorted(required_indexes)

    def _normalize_plan(self, plan: Plan) -> Plan:
        purposes_by_tool: dict[str, str] = {}
        for step in plan.steps:
            if step.tool in self.REQUIRED_PLAN_TOOLS and step.tool not in purposes_by_tool:
                purposes_by_tool[step.tool] = step.purpose

        normalized_steps = []
        for index, tool_name in enumerate(self.REQUIRED_PLAN_TOOLS, start=1):
            normalized_steps.append(
                PlanStep(
                    step_id=f"step_{index}",
                    tool=tool_name,
                    purpose=purposes_by_tool.get(tool_name, self.DEFAULT_PURPOSES[tool_name]),
                )
            )
        return Plan(summary=plan.summary, steps=normalized_steps)

    def _default_plan(self) -> Plan:
        return Plan(
            summary="Read the ticket, classify it, search the KB, draft a grounded reply, post the response, and update the ticket status.",
            steps=[
                PlanStep(step_id="step_1", tool="read_ticket", purpose=self.DEFAULT_PURPOSES["read_ticket"]),
                PlanStep(step_id="step_2", tool="classify_ticket", purpose=self.DEFAULT_PURPOSES["classify_ticket"]),
                PlanStep(step_id="step_3", tool="search_kb", purpose=self.DEFAULT_PURPOSES["search_kb"]),
                PlanStep(step_id="step_4", tool="draft_response", purpose=self.DEFAULT_PURPOSES["draft_response"]),
                PlanStep(step_id="step_5", tool="post_response", purpose=self.DEFAULT_PURPOSES["post_response"]),
                PlanStep(step_id="step_6", tool="update_ticket_status", purpose=self.DEFAULT_PURPOSES["update_ticket_status"]),
            ],
        )

    def _run_step(
        self,
        step: PlanStep,
        state: dict[str, object],
        ticket_id: str,
        log: ExecutionLog,
        conversation_turn: int,
    ) -> ToolOutcome | None:
        attempts_allowed = 1 + self.settings.max_retries_per_step
        last_outcome: ToolOutcome | None = None
        for attempt in range(1, attempts_allowed + 1):
            arguments = self._build_arguments(step.tool, ticket_id=ticket_id, state=state, attempt=attempt)
            if arguments is None:
                log.steps.append(
                    ExecutionStepRecord(
                        step_id=f"turn_{conversation_turn}_{step.step_id}",
                        tool=step.tool,
                        purpose=step.purpose,
                        conversation_turn=conversation_turn,
                        attempt=attempt,
                        arguments={},
                        outcome="skipped",
                        message="Step skipped because required context was not available.",
                        data={},
                        retryable=False,
                    )
                )
                return None
            outcome = self.toolkit.execute(step.tool, **arguments)
            last_outcome = outcome
            log.steps.append(
                ExecutionStepRecord(
                    step_id=f"turn_{conversation_turn}_{step.step_id}",
                    tool=step.tool,
                    purpose=step.purpose,
                    conversation_turn=conversation_turn,
                    attempt=attempt,
                    arguments=arguments,
                    outcome="success" if outcome.success else "failed",
                    message=outcome.message,
                    data=outcome.data,
                    retryable=outcome.retryable,
                )
            )
            if outcome.success:
                return outcome
            if not outcome.retryable or attempt >= attempts_allowed:
                return outcome
        return last_outcome

    def _build_arguments(
        self,
        tool_name: str,
        ticket_id: str,
        state: dict[str, object],
        attempt: int,
    ) -> dict[str, object] | None:
        ticket = self.repository.get(ticket_id)
        if tool_name == "read_ticket":
            return {"ticket_id": ticket_id}
        if tool_name == "classify_ticket":
            return {"ticket_id": ticket_id}
        if tool_name == "search_kb":
            classification = state.get("classification", {})
            issue_type = classification.get("issue_type") if isinstance(classification, dict) else None
            intent = classification.get("intent") if isinstance(classification, dict) else None
            classification_confidence = classification.get("confidence_score") if isinstance(classification, dict) else None
            return {
                "ticket_id": ticket_id,
                "query": self._build_search_query(
                    ticket.subject,
                    self.repository.current_customer_message(ticket_id),
                    issue_type,
                    intent,
                    attempt,
                ),
                "issue_type": issue_type,
                "intent": intent,
                "classification_confidence": classification_confidence,
                "limit": self.settings.search_results_limit,
            }
        if tool_name == "draft_response":
            kb_hits = state.get("kb_hits")
            if not isinstance(kb_hits, list) or not kb_hits:
                return None
            return {"ticket_id": ticket_id, "article_ids": [item["article_id"] for item in kb_hits]}
        if tool_name == "post_response":
            draft = state.get("draft_response")
            if not isinstance(draft, dict):
                return None
            return {
                "ticket_id": ticket_id,
                "response_body": draft["body"],
                "citations": draft.get("citations", []),
            }
        if tool_name == "update_ticket_status":
            draft = state.get("draft_response")
            if not isinstance(draft, dict):
                return None
            summary = str(state.get("status_summary") or draft["body"])[:180]
            return {
                "ticket_id": ticket_id,
                "status": str(state.get("next_status") or "resolved"),
                "resolution_summary": summary,
            }
        if tool_name == "escalate_ticket":
            reason = state.get("escalation_reason")
            if not isinstance(reason, str):
                return None
            return {"ticket_id": ticket_id, "reason": reason}
        return None

    def _build_search_query(self, subject: str, body: str, issue_type: str | None, intent: str | None, attempt: int) -> str:
        base = f"{subject} {body}".strip()
        if attempt == 1:
            return base
        if attempt == 2 and issue_type:
            parts = [item for item in [issue_type, intent, subject] if item]
            return " ".join(parts)
        words = [word for word in base.split() if len(word) > 4]
        trimmed = " ".join(words[:12]).strip()
        return trimmed or base

    def _apply_step_result(self, tool_name: str, outcome: ToolOutcome, state: dict[str, object]) -> None:
        if tool_name == "read_ticket":
            state["ticket"] = outcome.data.get("ticket", {})
        elif tool_name == "classify_ticket":
            state["classification"] = outcome.data
        elif tool_name == "search_kb":
            state["kb_search_context"] = outcome.data
            state["kb_hits"] = outcome.data.get("hits", [])
        elif tool_name == "draft_response":
            state["draft_response"] = outcome.data
            state["next_status"] = self._next_status_from_draft(outcome.data)
            state["status_summary"] = outcome.data.get("body", "")
            source = outcome.data.get("source")
            if isinstance(source, str):
                state["response_source"] = source
        elif tool_name == "post_response":
            state["posted_response"] = outcome.data
        elif tool_name == "update_ticket_status":
            state["status_update"] = outcome.data
        elif tool_name == "escalate_ticket":
            state["escalated"] = True
            customer_message = outcome.data.get("customer_message")
            if isinstance(customer_message, str):
                state["customer_escalation_message"] = customer_message

    def _escalate_after_failure(
        self,
        step: PlanStep,
        failure: ToolOutcome,
        state: dict[str, object],
        ticket_id: str,
        log: ExecutionLog,
        conversation_turn: int,
    ) -> bool:
        reason = self._build_escalation_reason(
            step=step,
            failure=failure,
            state=state,
            ticket_id=ticket_id,
            log=log,
            conversation_turn=conversation_turn,
        )
        state["escalation_reason"] = reason
        arguments = {"ticket_id": ticket_id, "reason": reason}
        outcome = self.toolkit.execute("escalate_ticket", **arguments)
        log.steps.append(
            ExecutionStepRecord(
                step_id=f"turn_{conversation_turn}_{step.step_id}_escalation",
                tool="escalate_ticket",
                purpose=f"Escalate because {step.tool} could not be completed.",
                conversation_turn=conversation_turn,
                attempt=1,
                arguments=arguments,
                outcome="escalated" if outcome.success else "failed",
                message=outcome.message,
                data=outcome.data,
                retryable=outcome.retryable,
            )
        )
        if outcome.success:
            self._apply_step_result("escalate_ticket", outcome, state)
            return True
        return False

    def _build_escalation_reason(
        self,
        step: PlanStep,
        failure: ToolOutcome,
        state: dict[str, object],
        ticket_id: str,
        log: ExecutionLog,
        conversation_turn: int,
    ) -> str:
        parts = ["Escalation required."]
        classification = state.get("classification")
        if isinstance(classification, dict):
            issue_type = classification.get("issue_type") or "unknown"
            intent = classification.get("intent") or "unknown"
            confidence = classification.get("confidence_score")
            if isinstance(confidence, (int, float)):
                parts.append(f"Classification: {issue_type} / {intent} ({confidence:.2f} confidence).")
            else:
                parts.append(f"Classification: {issue_type} / {intent}.")

        current_message = self.repository.current_customer_message(ticket_id).strip()
        if current_message:
            parts.append(f'Latest customer message: "{current_message[:220]}".')

        kb_hits = state.get("kb_hits")
        if isinstance(kb_hits, list) and kb_hits:
            evidence_fragments: list[str] = []
            for hit in kb_hits[:3]:
                if not isinstance(hit, dict):
                    continue
                article_id = str(hit.get("article_id", "unknown"))
                matched_terms = hit.get("matched_terms", [])
                if isinstance(matched_terms, list) and matched_terms:
                    evidence_fragments.append(f"{article_id} matched {', '.join(str(term) for term in matched_terms[:4])}")
                else:
                    evidence_fragments.append(article_id)
            if evidence_fragments:
                parts.append("Evidence reviewed: " + "; ".join(evidence_fragments) + ".")

        search_attempts = [
            record
            for record in log.steps
            if record.tool == "search_kb" and record.conversation_turn == conversation_turn
        ]
        if search_attempts:
            attempt_fragments: list[str] = []
            for record in search_attempts[-3:]:
                query = str(record.arguments.get("query", "")).strip()
                if query:
                    attempt_fragments.append(f'"{query[:120]}" -> {record.message}')
            if attempt_fragments:
                parts.append("KB attempts: " + " | ".join(attempt_fragments) + ".")

        posted_responses = [
            record
            for record in log.steps
            if record.tool == "post_response" and record.outcome == "success"
        ]
        if posted_responses:
            latest_guidance = str(posted_responses[-1].arguments.get("response_body", "")).strip()
            if latest_guidance:
                parts.append(f'Latest agent guidance: "{latest_guidance[:220]}".')

        failure_body = failure.data.get("body")
        if isinstance(failure_body, str) and failure_body.strip():
            parts.append(f'Draft assessment: "{failure_body[:220]}".')

        parts.append(f"Escalation trigger: {step.tool} failed with '{failure.message}'.")
        return " ".join(parts)

    def _extract_final_response(self, state: dict[str, object]) -> str | None:
        draft = state.get("draft_response")
        if isinstance(draft, dict):
            return draft.get("body")
        escalation_reason = state.get("escalation_reason")
        if isinstance(escalation_reason, str):
            return escalation_reason
        return None

    def _extract_response_source(self, state: dict[str, object]) -> str:
        source = state.get("response_source")
        if isinstance(source, str):
            return source
        return "none"

    def _next_status_from_draft(self, draft: dict[str, object]) -> str:
        next_action = draft.get("next_action")
        if next_action == "resolve":
            return "resolved"
        if next_action == "await_customer":
            return "pending_customer"
        return "open"

    def _consume_customer_reply(self, ticket_id: str, log: ExecutionLog, conversation_turn: int) -> bool:
        reply = self.repository.consume_next_customer_reply(ticket_id)
        if reply is None:
            return False
        log.steps.append(
            ExecutionStepRecord(
                step_id=f"turn_{conversation_turn}_customer_reply",
                tool="customer_reply",
                purpose="Receive the next customer message and continue the workflow.",
                conversation_turn=conversation_turn,
                attempt=1,
                arguments={"ticket_id": ticket_id},
                outcome="success",
                message="Customer replied and the workflow continued.",
                data={"message": reply},
                retryable=False,
            )
        )
        return True

    def _reset_turn_state(self, state: dict[str, object]) -> None:
        for key in [
            "ticket",
            "classification",
            "kb_hits",
            "kb_search_context",
            "draft_response",
            "posted_response",
            "status_update",
            "next_status",
            "status_summary",
        ]:
            state.pop(key, None)

    def _escalate_due_to_turn_limit(self, ticket_id: str, state: dict[str, object], log: ExecutionLog) -> bool:
        parts = [
            "Escalation required.",
            f"Conversation exceeded the maximum supported turns ({self.settings.max_conversation_turns}).",
        ]
        classification = state.get("classification")
        if isinstance(classification, dict):
            issue_type = classification.get("issue_type") or "unknown"
            intent = classification.get("intent") or "unknown"
            parts.append(f"Latest classification: {issue_type} / {intent}.")
        current_message = self.repository.current_customer_message(ticket_id).strip()
        if current_message:
            parts.append(f'Latest customer message: "{current_message[:220]}".')
        reason = " ".join(parts)
        state["escalation_reason"] = reason
        arguments = {"ticket_id": ticket_id, "reason": reason}
        outcome = self.toolkit.execute("escalate_ticket", **arguments)
        log.steps.append(
            ExecutionStepRecord(
                step_id=f"turn_{self.settings.max_conversation_turns}_turn_limit_escalation",
                tool="escalate_ticket",
                purpose="Escalate because the conversation exceeded the supported turn limit.",
                conversation_turn=self.settings.max_conversation_turns,
                attempt=1,
                arguments=arguments,
                outcome="escalated" if outcome.success else "failed",
                message=outcome.message,
                data=outcome.data,
                retryable=outcome.retryable,
            )
        )
        if outcome.success:
            self._apply_step_result("escalate_ticket", outcome, state)
            return True
        return False
