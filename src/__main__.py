import logging
import os

import click
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from dotenv import load_dotenv
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from .nvidia_agent import create_agent
from .nvidia_agent_executor import NvidiaAgentExecutor
from starlette.applications import Starlette

load_dotenv()
logging.basicConfig()


@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=5000)
def main(host: str, port: int):
    if not os.getenv("NVIDIA_API_KEY"):
        raise ValueError("NVIDIA_API_KEY environment variable not set")

    skill = AgentSkill(
        id="customer_ticket_resolution",
        name="Customer Ticket Resolution",
        description="PS3 Domain 1 workflow orchestration with RAG, retries, and escalation.",
        tags=["customer-support", "workflow", "orchestration", "rag", "escalation"],
        examples=[
            "Customer says their earbuds are not pairing over bluetooth.",
            "Customer reports duplicate posted charge and requests help.",
            "Customer asks for delayed shipment resolution and next actions.",
        ],
    )

    agent_card = AgentCard(
        name="battleborn-customer-support-ai",
        description="PS3 Domain 1 Customer Ticket Resolution Agent.",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    agent_data = create_agent()
    agent_executor = NvidiaAgentExecutor(
        card=agent_card,
        tools=agent_data["tools"],
        api_key=os.getenv("NVIDIA_API_KEY"),
        system_prompt=agent_data["system_prompt"],
    )
    workflow_tool = agent_data["workflow"]

    async def plain_message(request: Request) -> JSONResponse:
        payload = await request.json()
        message = str(payload.get("message", "")).strip()
        if not message:
            return JSONResponse({"error": "message is required"}, status_code=400)
        result = await workflow_tool.run_ticket_workflow(message)
        return JSONResponse(result)

    request_handler = DefaultRequestHandler(agent_executor=agent_executor, task_store=InMemoryTaskStore())
    a2a_app = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
    routes = [Route("/agent/message", endpoint=plain_message, methods=["POST"]), *a2a_app.routes()]
    app = Starlette(routes=routes)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

