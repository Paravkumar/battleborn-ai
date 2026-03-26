from __future__ import annotations

import logging
import os

import click
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from dotenv import load_dotenv
from starlette.applications import Starlette

from a2a_agent.openai_agent import create_agent
from a2a_agent.openai_agent_executor import OpenAIAgentExecutor

load_dotenv()
logging.basicConfig()


@click.command()
@click.option("--host", default="localhost")
@click.option("--port", default=5000, type=int)
def main(host: str, port: int) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")

    skill = AgentSkill(
        id="ticket_resolution",
        name="Ticket Resolution",
        description="Resolves Battleborn customer-support tickets using the existing ticket-agent runtime.",
        tags=["support", "ticket", "battleborn", "resolution"],
        examples=["Resolve ticket TICK-1001", "Run support workflow for ticket TICK-1020"],
    )

    card = AgentCard(
        name="battleborn-customer-support-ai",
        description="A2A wrapper around the Battleborn ticket resolution agent.",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    agent_data = create_agent()
    executor = OpenAIAgentExecutor(
        card=card,
        tools=agent_data["tools"],
        api_key=os.getenv("OPENAI_API_KEY", ""),
        system_prompt=str(agent_data["system_prompt"]),
    )
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    app = Starlette(routes=A2AStarletteApplication(agent_card=card, http_handler=handler).routes())
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

