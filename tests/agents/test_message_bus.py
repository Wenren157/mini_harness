import pytest

from mini_harness.agents.message_bus import (
    MessageBus,
    AgentMessage
)

@pytest.mark.asyncio
async def test_agent_message_bus_transfer():

    bus = MessageBus()
    bus.register_agent("executor")

    msg = AgentMessage(
        sender="planner",
        receiver="executor",
        content="execute step 1"
    )
    await bus.send(msg)
    received = await bus.receive("executor")

    assert (received.sender == "planner")

    assert (received.content == "execute step 1" )

@pytest.mark.asyncio
async def test_agent_message_isolation():
    bus = MessageBus()
    bus.register_agent("agent_a")
    bus.register_agent("agent_b")

    await bus.send(
        AgentMessage(
            sender="agent_a",
            receiver="agent_b",
            content="hello"
        )
    )

    msg = await bus.receive("agent_b")
    assert (msg.receiver =="agent_b")
    assert (msg.content =="hello")