import asyncio
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.harness.business_agent_executor import AgentRunResult, BusinessAgentExecutor
from app.harness.customer_service_harness import CustomerServiceHarness
from app.state.request_models import (
    CustomerProfile,
    CustomerServiceRequest,
    SlotDefinition,
)


async def fake_runner(
    state,
    instructions,
    user_message,
    tools,
    mcp_servers,
    max_turns,
    model,
):
    del instructions, user_message, tools, mcp_servers, max_turns, model
    if "human" in state.content.lower():
        state.need_handoff_to_human = True
        state.handoff_type = "reply_handoff"
        state.handoff_reason = "customer requested human support"

    customer_name = state.customer.name or "customer"
    return AgentRunResult(
        final_output=(
            f"Dear {customer_name},\n\n"
            f"We can help with your request: {state.content}\n\n"
            "Best regards,\nSupport Team"
        ),
        token_usage={
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    )


async def main() -> None:
    harness = CustomerServiceHarness(
        executor=BusinessAgentExecutor(runner=fake_runner)
    )
    cases = [
        CustomerServiceRequest(
            request_id="demo_order",
            tenant_id="tenant_a",
            channel="email",
            subject="Order status",
            content="Where is my order A100?",
            customer=CustomerProfile(name="Renée", id="user_renee"),
            slot_schema=[
                SlotDefinition(name="order_id", description="Order id")
            ],
        ),
        CustomerServiceRequest(
            request_id="demo_handoff",
            tenant_id="tenant_a",
            channel="chat",
            content="I want to talk to a human agent.",
            customer=CustomerProfile(name="Sam"),
        ),
    ]

    # One compact JSON object per line keeps subprocess output easy to stream and parse.
    for case in cases:
        response = await harness.run(case)
        print(
            json.dumps(
                response.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
