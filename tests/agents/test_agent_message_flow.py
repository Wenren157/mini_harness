"""
测试 Multi-Agent 基础通信流程

验证：

Agent A
 |
Message
 |
Agent B

目标：
- Agent之间可以传递任务信息
- 不依赖 Runtime
- 不依赖 ToolRegistry
- 不污染 Runtime状态
"""

import pytest
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class AgentMessage:
    """
    Agent之间通信的数据结构
    """
    sender: str
    receiver: str
    content: Dict[str, Any]

class MockAgent:
    """
    最小Agent模型

    注意：
    这里只模拟通信能力，
    不包含Runtime执行能力。
    """
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.received_messages = []

    async def send_message(
        self,
        target,
        message: AgentMessage
    ):
        await target.receive_message(message)

    async def receive_message(
        self,
        message: AgentMessage
    ):
        self.received_messages.append(message)

class TestAgentMessageFlow:
    @pytest.mark.asyncio
    async def test_agent_message_transfer(self):
        """
        Agent A发送任务给Agent B

        验证：
        B成功收到消息
        """
        planner = MockAgent("planner")
        executor = MockAgent("executor")

        message = AgentMessage(
            sender="planner",
            receiver="executor",
            content={
                "task": "execute read_file runtime.py"
            }
        )

        await planner.send_message(
            executor,
            message
        )

        assert len(executor.received_messages) == 1

        received = (executor.received_messages[0])

        assert received.sender == "planner"
        assert (received.content["task"]=="execute read_file runtime.py")

    @pytest.mark.asyncio
    async def test_agent_message_isolation(self):
        """
        验证：Agent A消息不会污染Agent B之外的数据
        """
        agent_a = MockAgent("agent_a")
        agent_b = MockAgent("agent_b")
        agent_c = MockAgent("agent_c")

        message = AgentMessage(
            sender="agent_a",
            receiver="agent_b",
            content={
                "data":"hello"
            }
        )

        await agent_a.send_message(
            agent_b,
            message
        )

        assert len(agent_b.received_messages) == 1
        assert len(agent_c.received_messages) == 0

    def test_message_contains_agent_identity(self):
        """
        验证消息必须携带身份信息,为未来MessageBus做准备
        """
        message = AgentMessage(
            sender="planner",
            receiver="executor",
            content={}
        )

        assert message.sender == "planner"
        assert message.receiver == "executor"
