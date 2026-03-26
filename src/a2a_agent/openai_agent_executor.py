from __future__ import annotations

import inspect
import json
import logging
from typing import Any

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import AgentCard, TaskState, TextPart, UnsupportedOperationError
from a2a.utils.errors import ServerError
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class OpenAIAgentExecutor(AgentExecutor):
    """Runs the OpenAI-powered A2A agent with function calling."""

    def __init__(self, card: AgentCard, tools: dict[str, Any], api_key: str, system_prompt: str):
        self._card = card
        self.tools = tools
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o"
        self.system_prompt = system_prompt

    async def _process_request(self, message_text: str, task_updater: TaskUpdater) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": message_text},
        ]

        openai_tools: list[dict[str, Any]] = []
        for tool_name, tool_instance in self.tools.items():
            if hasattr(tool_instance, tool_name):
                openai_tools.append(
                    {
                        "type": "function",
                        "function": self._extract_function_schema(getattr(tool_instance, tool_name)),
                    }
                )

        max_iterations = 10
        for _ in range(max_iterations):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=openai_tools or None,
                    tool_choice="auto" if openai_tools else None,
                    temperature=0.1,
                    max_tokens=3000,
                )
                message = response.choices[0].message
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": message.tool_calls,
                    }
                )

                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)

                        if function_name in self.tools and hasattr(self.tools[function_name], function_name):
                            method = getattr(self.tools[function_name], function_name)
                            result = method(**function_args)
                            if inspect.iscoroutine(result):
                                result = await result
                        else:
                            result = {"error": f"Function {function_name} not found"}

                        if hasattr(result, "model_dump"):
                            result_json = json.dumps(result.model_dump())
                        elif isinstance(result, dict):
                            result_json = json.dumps(result)
                        else:
                            result_json = str(result)

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result_json,
                            }
                        )

                    await task_updater.update_status(
                        TaskState.working,
                        message=task_updater.new_agent_message([TextPart(text="Processing tool calls...")]),
                    )
                    continue

                if message.content:
                    await task_updater.add_artifact([TextPart(text=message.content)])
                await task_updater.complete()
                return
            except Exception as exc:
                await task_updater.add_artifact([TextPart(text=f"Request failed: {exc!s}")])
                await task_updater.complete()
                return

        await task_updater.add_artifact([TextPart(text="Max iterations reached.")])
        await task_updater.complete()

    def _extract_function_schema(self, func: Any) -> dict[str, Any]:
        sig = inspect.signature(func)
        description = (inspect.getdoc(func) or "").split("\n")[0] or func.__name__
        properties: dict[str, dict[str, str]] = {}
        required: list[str] = []
        for param_name, param in sig.parameters.items():
            param_type = "string"
            if param.annotation == int:
                param_type = "integer"
            elif param.annotation == float:
                param_type = "number"
            elif param.annotation == bool:
                param_type = "boolean"
            elif param.annotation == list:
                param_type = "array"
            elif param.annotation == dict:
                param_type = "object"
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
            properties[param_name] = {"type": param_type, "description": f"Parameter {param_name}"}
        return {
            "name": func.__name__,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        }

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            await updater.submit()
        await updater.start_work()

        message_text = ""
        for part in context.message.parts:
            if isinstance(part.root, TextPart):
                message_text += part.root.text
        await self._process_request(message_text, updater)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())

