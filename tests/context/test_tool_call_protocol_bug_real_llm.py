import os
from types import SimpleNamespace

import pytest

from mini_harness.core.runtime import HarnessRuntime
from mini_harness.core.models import AgentStatus
from mini_harness.infra.real_llm_client import RealLLMClient


class MinimalToolRegistry:
    """
    Protocol Bug Test 专用最小 ToolRegistry。

    本测试不测试 ToolRegistry，
    只测试：

        Context compression
            ↓
        Runtime
            ↓
        RealLLMClient
            ↓
        DeepSeek API

    是否会因为 Tool Calling 消息被拆散而产生协议错误。
    """

    def get_schema(self):
        return None

    async def execute_with_retry(self, name, **kwargs):
        raise AssertionError(
            "Protocol Bug Test 不应该真正执行工具"
        )


@pytest.mark.asyncio
async def test_context_compression_must_not_break_tool_call_protocol():
    """
    P0 regression test.

    目标：
        从一个合法的 Tool Calling Context 开始，
        经过 ContextManager 自己的真实 compression，
        再由 Runtime 发送给真实 DeepSeek API。

    历史 Bug：
        旧 compression 按 message 数量执行：
            kept = non_system[-3:]

        因而可能把：
            assistant(tool_calls)
            tool

        拆成：
            tool

        最终制造 orphan tool message，导致 DeepSeek 返回：
            Messages with role 'tool' must be a response
            to a preceding message with 'tool_calls'

    当前长期 invariant：
        1. Compression 必须真实发生；
        2. old history -> summary；
        3. recent messages -> raw；
        4. assistant(tool_calls) + 对应 tool result(s)必须作为不可拆分的 Message Block；
        5. Runtime -> RealLLMClient -> DeepSeek不应产生 Tool Calling protocol error。

    注意：
        本测试不实现 Protocol Guard，
        也不人工制造非法 Context。
    """

    # ============================================================
    # 1. 必须使用真实 DeepSeek API
    # ============================================================
    assert os.getenv("DEEPSEEK_API_KEY"), (
        "Protocol Bug Test 必须调用真实 DeepSeek API。\n"
        "请先设置环境变量 DEEPSEEK_API_KEY。"
    )
    llm = RealLLMClient()

    # ============================================================
    # 2. 创建最小 Runtime
    # ============================================================
    config = SimpleNamespace(
        event_bus_maxlen=100,
        workspace=".",
        max_context_tokens=8000,
        max_iterations=1,
        enable_mcp=False,
        tool_timeout=5.0,
        enable_concurrent_tools=False,
    )

    runtime = HarnessRuntime(
        config=config,
        llm_client=llm,
        tool_registry=MinimalToolRegistry(),
    )

    # ============================================================
    # 3. 构造一个“最开始完全合法”的 Tool Calling Context
    #
    # Message Block 结构：
    #
    #   Block 1:
    #       user(old history)
    #
    #   Block 2:
    #       assistant(tool_calls)
    #       tool
    #
    #   Block 3:
    #       user
    #
    #   Block 4:
    #       user
    #
    # P0-2 compression 保留最近 3 个 block：
    #
    #   Block 1 -> summary
    #
    #   Block 2 / 3 / 4 -> raw
    #
    # 因此可以真实验证：
    # compression boundary 不得拆开 Tool Calling block。
    # ============================================================

    runtime.context.add_user_message("old history that should be compressed")
    runtime.context.add_assistant_tool_calls(
        [
            {
                "id": "call_protocol_bug",
                "type": "function",
                "function": {
                    "name": "dummy_tool",
                    "arguments": "{}",
                },
            }
        ]
    )

    runtime.context.add_tool_result(
        "call_protocol_bug",
        "protocol bug reproduction result",
    )

    runtime.context.add_user_message(
        "compression filler message one"
    )

    runtime.context.add_user_message(
        "compression filler message two"
    )

    # ============================================================
    # 4. 首先证明：
    #
    # compression 之前 Context 是合法的。
    #
    # 这是本测试和上一版最重要的区别。
    # ============================================================

    before_compression = runtime.context.window.messages.copy()

    assert len(before_compression) == 5
    assert before_compression[0]["role"] == "user"
    assert before_compression[1]["role"] == "assistant"
    assert before_compression[1].get("tool_calls"), (
        "测试前置条件错误："
        "assistant message 必须包含 tool_calls"
    )
    assert before_compression[2]["role"] == "tool"
    assert (
        before_compression[2]["tool_call_id"]
        == before_compression[1]["tool_calls"][0]["id"]
    ), (
        "测试前置条件错误："
        "tool_call_id 必须和 assistant tool_calls id 对应"
    )
    tool_call = before_compression[1]["tool_calls"][0]
    assert tool_call["type"] == "function", (
        "测试前置条件错误："
        "provider tool_call 必须包含 type='function'"
    )
    assert tool_call["function"]["name"] == "dummy_tool", (
        "测试前置条件错误："
        "provider tool_call 必须包含 function.name"
    )
    assert isinstance(
        tool_call["function"]["arguments"],
        str,
    ), (
        "测试前置条件错误："
        "provider tool_call function.arguments 必须是字符串"
    )

    print(
        "\n========== BEFORE COMPRESSION =========="
    )

    for index, message in enumerate(before_compression):
        print(
            index,
            message.get("role"),
            "tool_calls=",
            message.get("tool_calls"),
            "tool_call_id=",
            message.get("tool_call_id"),
        )

    print(
        "========================================\n"
    )

    # ============================================================
    # 5. 调用 ContextManager 自己真实存在的 compression
    #
    # 不手工删消息。
    # 不手工制造 orphan tool。
    #
    # _compress_background() 内部本身还会调用：
    #
    #     self.llm.generate(summary_messages)
    #
    # 因此这里同样使用的是真实 RealLLMClient / DeepSeek API。
    # ============================================================

    await runtime.context._compress_background()

    after_compression = (runtime.context.window.messages.copy())
    print(
        "\n========== AFTER COMPRESSION =========="
    )

    # ============================================================
    # 6. 证明 compression 真的发生过，并且没有拆散
    # Tool Calling Message Block。
    # ============================================================

    assert len(after_compression) == 5, (
        "compression 后应为："
        "summary + assistant(tool_calls) + tool + user + user"
    )

    # Block 1 的 old history 应已经被压缩成 summary。
    assert after_compression[0]["role"] == "system"
    assert "[历史摘要]" in after_compression[0]["content"]

    assert all(
        message.get("content")
        != "old history that should be compressed"
        for message in after_compression
    ), (
        "old history 应该已经进入 summary，"
        "不应继续作为 raw message 保留"
    )

    # Tool Calling Block 必须完整保留。
    assert after_compression[1]["role"] == "assistant"
    assert after_compression[1].get("tool_calls")
    assert after_compression[2]["role"] == "tool"
    assert (
        after_compression[2]["tool_call_id"]
        == after_compression[1]["tool_calls"][0]["id"]
    ), (
        "compression 后 Tool Calling Message Block 被拆坏"
    )

    for index, message in enumerate(after_compression):
        print(
            index,
            message.get("role"),
            "tool_calls=",
            message.get("tool_calls"),
            "tool_call_id=",
            message.get("tool_call_id"),
        )

    print(
        "=======================================\n"
    )

    # ============================================================
    # 7. 继续走真实 Runtime
    #
    # run() 会追加新的 user message，
    # 然后：
    #
    # Context
    #   ↓
    # get_context_for_llm()
    #   ↓
    # Runtime._step()
    #   ↓
    # RealLLMClient.generate()
    #   ↓
    # DeepSeek API
    #
    # 历史 Bug 下，DeepSeek 会看到 orphan tool。
    #
    # P0-2 修复后：
    # Runtime 应发送完整的
    # assistant(tool_calls) -> tool
    # 协议消息组。
    # ============================================================

    result = await runtime.run(
        "请简单回复 protocol regression test passed。"
    )

    # ============================================================
    # 8. 最终长期回归断言
    #
    # Context compression 不允许破坏
    # Tool Calling protocol。
    #
    # 因此 Runtime 不应该收到这个 provider protocol error。
    #
    # 历史版本：
    #     RED
    #
    # P0-2 修复后：
    #     GREEN
    # ============================================================

    error_info = runtime.state.error_info or ""
    protocol_error = (
        "Messages with role 'tool' must be a response "
        "to a preceding message with 'tool_calls'"
    )

    assert protocol_error not in error_info, (
        "\n"
        "Context compression broke Tool Calling protocol.\n\n"
        "测试初始 Context 是合法的：\n"
        "assistant(tool_calls) -> tool\n\n"
        "但 compression 后 Runtime 向真实 DeepSeek API "
        "发送了 orphan tool message。\n\n"
        f"Runtime error_info:\n{error_info}\n"
    )

    assert result["status"] != AgentStatus.ERROR, (
        "Runtime 在 Context compression "
        "Protocol Bug Test 中进入 ERROR：\n"
        f"{error_info}"
    )