import pytest
import asyncio

from mini_harness.core.runtime import HarnessRuntime
from mini_harness.infra.tools import ToolRegistry, SandboxExecutor
from mini_harness.infra.config import RuntimeConfig
from tests.mocks import MockLLMClient


@pytest.mark.asyncio
async def test_context_manager_message_migration():
    """
    验证事务2：
    Runtime消息已经从 state.messages 迁移到 ContextManager

    验证：
    1. 用户消息进入 ContextManager
    2. LLM读取ContextManager
    3. 工具结果进入ContextManager
    """

    config = RuntimeConfig(
        max_iterations=2
    )

    llm = MockLLMClient(mode="tool")

    sandbox = SandboxExecutor()

    registry = ToolRegistry(
        sandbox=sandbox
    )


    async def add_func(a, b):
        return a + b


    registry.register(
        "add",
        add_func,
        "add numbers",
        {
            "a": "int",
            "b": "int"
        }
    )


    runtime = HarnessRuntime(
        config=config,
        llm_client=llm,
        tool_registry=registry
    )


    result = await runtime.run(
        "What is 1+2?"
    )


    # =========================
    # 1. 验证 ContextManager 收到了用户消息
    # =========================

    context_messages = runtime.context.get_context_for_llm()

    assert any(
        msg["role"] == "user"
        and msg["content"] == "What is 1+2?"
        for msg in context_messages
    ), "用户消息没有进入 ContextManager"


    # =========================
    # 2. 验证 LLM 实际收到的是 ContextManager 的消息
    # =========================

    assert len(llm.call_history) >= 1


    first_llm_messages = (
        llm.call_history[0]["messages"]
    )


    assert any(
        msg["role"] == "user"
        and msg["content"] == "What is 1+2?"
        for msg in first_llm_messages
    ), "第一次LLM调用没有拿到ContextManager中的用户消息"

    assert len(llm.call_history) >= 2
    second_llm_messages = llm.call_history[1]["messages"]


    assert any(
        msg["role"] == "tool"
        for msg in second_llm_messages
    ), "第二次LLM调用没有拿到工具结果"

    # =========================
    # 3. 验证工具结果进入 ContextManager
    # =========================

    context_messages = runtime.context.get_context_for_llm()


    assert any(
        msg["role"] == "tool"
        for msg in context_messages
    ), "工具结果没有进入ContextManager"


    # =========================
    # 4. 验证 state.messages 没有被继续使用
    # =========================

    assert len(runtime.state.messages) == 0, \
        "Runtime仍然在使用state.messages"

