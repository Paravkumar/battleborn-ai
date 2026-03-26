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
    def __init__(self, card: AgentCard, tools: dict[str, Any], api_key: str, system_prompt: str):
        self._card = card
        self.tools = tools
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4.1-mini"
        self.system_prompt = system_prompt

    def _extract_function_schema(self, func: Any) -> dict[str, Any]:
        sig = inspect.signature(func)
        description = (inspect.getdoc(func) or "").split("\n")[0] or func.__name__
        properties = {}
        required = []
        for name, param in sig.parameters.items():
            if param.default == inspect.Parameter.empty:
                required.append(name)
            properties[name] = {"type": "string", "description": f"Parameter {name}"}
        return {
            "name": func.__name__,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        }

    async def _process_request(self, message_text: str, task_updater: TaskUpdater) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": message_text},
        ]

        openai_tools = []
        for tool_name, tool_instance in self.tools.items():
            if hasattr(tool_instance, tool_name):
                openai_tools.append(
                    {"type": "function", "function": self._extract_function_schema(getattr(tool_instance, tool_name))}
                )

        for _ in range(10):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=openai_tools if openai_tools else None,
                tool_choice="auto" if openai_tools else None,
                temperature=0.2,
                max_tokens=2500,
            )
            message = response.choices[0].message
            messages.append({"role": "assistant", "content": message.content, "tool_calls": message.tool_calls})

            if not message.tool_calls:
                if message.content:
                    await task_updater.add_artifact([TextPart(text=message.content)])
                await task_updater.complete()
                return

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
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result) if isinstance(result, dict) else str(result),
                    }
                )
            await task_updater.update_status(
                TaskState.working, message=task_updater.new_agent_message([TextPart(text="Processing...")])
            )

        await task_updater.add_artifact([TextPart(text="Maximum iterations reached.")])
        await task_updater.complete()

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            await updater.submit()
        await updater.start_work()
        message_text = ""
        for part in context.message.parts:
            if isinstance(part.root, TextPart):
                message_text += part.root.text
        await self._process_request(message_text, updater)

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise ServerError(error=UnsupportedOperationError())

