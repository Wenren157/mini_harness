import asyncio
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class AgentMessage:
    """
    Agent之间通信消息
    """
    sender: str
    receiver: str
    content: Any
    metadata: Dict[str, Any] | None = None


class MessageBus:
    """
    Multi-Agent通信总线

    当前实现:
    - 内存队列
    - agent隔离队列

    后续可扩展:
    - Redis
    - Kafka
    - MCP Message Transport
    """
    def __init__(self):
        self.queues: Dict[str, asyncio.Queue] = {}

    def register_agent(
        self,
        agent_id: str
    ):
        """
        注册Agent
        """
        if agent_id not in self.queues:
            self.queues[agent_id] = (asyncio.Queue())

    async def send(
        self,
        message: AgentMessage
    ):
        """
        发送消息
        """
        if (
            message.receiver
            not in self.queues
        ):
            raise ValueError(f"Unknown agent: {message.receiver}")

        await self.queues[
            message.receiver
        ].put(message)

    async def receive(
        self,
        agent_id: str
    ) -> AgentMessage:
        """
        接收消息
        """
        if agent_id not in self.queues:
            raise ValueError(f"Unknown agent: {agent_id}")

        return await self.queues[
            agent_id
        ].get()