"""
P0-2 Context Compression Regression Tests
验证两个 Context correctness invariant：
    1. Compression 应压缩旧历史，而不是总结近期消息后
    又把近期消息原样保留。

    2. assistant(tool_calls) + 对应 tool result(s)
    必须作为一个不可拆分的协议单元保留。

注意：
这里是 Context 层单元测试，不调用真实 DeepSeek API。
真实 Provider 级回归由：
    tests/runtime/test_tool_call_protocol_bug_real_llm.py
负责。
"""

import pytest
from mini_harness.infra.context import ContextManager

class RecordingLLM:
    """
    Compression 测试专用最小 LLM。
    只负责两件事：
    1. 记录 ContextManager 发给 summarizer 的 messages；
    2. 返回固定 summary。
    它不参与 Tool Calling，也不模拟 Runtime 行为。
    """
    def __init__(self):
        self.requests = []

    async def generate(
        self,
        messages,
        tools=None,
    ):
        self.requests.append(messages)
        return {
            "content": "compressed history",
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": None,
        }

@pytest.mark.asyncio
async def test_compression_summarizes_old_history_and_keeps_recent_messages():
    """
    P0-2 invariant 1：
    Compression 应：

        old history
            ↓
        summary
    而：
        recent messages
            ↓
        保持原文

    当前旧实现：
        recent_5 = messages_to_summarize[-5:]
        kept = non_system[-3:]

    会导致最近 3 条既进入 summary，
    又作为原文继续保留。
    当前代码应 RED。
    """

    llm = RecordingLLM()
    context = ContextManager(
        llm_client=llm,
        max_tokens=100_000,
    )

    # ------------------------------------------------------------
    # 构造 5 个普通 message block
    # 正确的未来分区：
    # history_to_summarize:
    #     history-1
    #     history-2
    # kept:
    #     recent-1
    #     recent-2
    #     recent-3
    # ------------------------------------------------------------

    context.add_user_message("history-1")
    context.add_assistant_message("history-2")
    context.add_user_message("recent-1")
    context.add_assistant_message("recent-2")
    context.add_user_message("recent-3")

    await context._compress_background()

    # Compression 应只调用一次 summarizer。
    assert len(llm.requests) == 1

    summary_request = llm.requests[0]

    # 当前 compression 使用：
    #
    # [
    #   system(summary_prompt),
    #   user(str(history))
    # ]
    #
    # 所以检查第二条 user message 即可。
    summary_source = summary_request[1]["content"]

    # ------------------------------------------------------------
    # 旧历史必须进入 summary
    # ------------------------------------------------------------
    assert "history-1" in summary_source
    assert "history-2" in summary_source

    # ------------------------------------------------------------
    # 最近消息必须原样保留，
    # 不应该重复进入 summary。
    #
    # 当前实现会在这里 FAIL。
    # ------------------------------------------------------------
    assert "recent-1" not in summary_source
    assert "recent-2" not in summary_source
    assert "recent-3" not in summary_source


@pytest.mark.asyncio
async def test_compression_preserves_tool_call_message_block():
    """
    P0-2 invariant 2：
    assistant(tool_calls)
            +
    tool(result)
    必须作为一个 Message Block,Compression 边界不能切在两者之间。

    初始合法 Context：
        old-history
        assistant(tool_calls=call_001)
        tool(tool_call_id=call_001)

        recent-user-1
        recent-user-2

    从 block 角度：
        Block 1 = old-history
        Block 2 =
            assistant(tool_calls)
            tool
        Block 3 = recent-user-1
        Block 4 = recent-user-2

    保留最近 3 个 block 后应得到：
        assistant(tool_calls)
        tool
        recent-user-1
        recent-user-2

    当前旧实现：
        kept = non_system[-3:]

    会得到：
        tool
        recent-user-1
        recent-user-2
    从而产生 orphan tool,当前代码应 RED。
    """

    llm = RecordingLLM()
    context = ContextManager(
        llm_client=llm,
        max_tokens=100_000,
    )

    # ------------------------------------------------------------
    # Block 1：旧历史
    # ------------------------------------------------------------
    context.add_user_message(
        "old-history"
    )

    # ------------------------------------------------------------
    # Block 2：合法 Tool Calling 协议组
    #
    # 使用 Runtime 实际发送给 Context 的 OpenAI-compatible
    # assistant tool_calls 结构。
    # ------------------------------------------------------------
    context.add_assistant_tool_calls(
        [
            {
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "dummy_tool",
                    "arguments": "{}",
                },
            }
        ]
    )
    context.add_tool_result(
        tool_call_id="call_001",
        result="tool-result",
    )

    # ------------------------------------------------------------
    # Block 3 / Block 4：近期普通消息
    # ------------------------------------------------------------
    context.add_user_message("recent-user-1")
    context.add_user_message("recent-user-2")

    # ------------------------------------------------------------
    # 首先证明测试输入本身是合法的。
    # ------------------------------------------------------------
    before = context.window.messages.copy()

    assert before[1]["role"] == "assistant"
    assert before[1].get("tool_calls")
    assert before[2]["role"] == "tool"
    assert (
        before[1]["tool_calls"][0]["id"]
        == before[2]["tool_call_id"]
    )

    # ------------------------------------------------------------
    # 执行真正的 Context compression。
    # ------------------------------------------------------------
    await context._compress_background()
    after = context.window.messages.copy()

    # ------------------------------------------------------------
    # 找到 compression 后保留下来的 tool。
    #
    # 当前实现中它仍然存在，但 assistant(tool_calls)
    # 会因为 [-3:] 边界被删掉。
    # ------------------------------------------------------------
    tool_messages = [
        message
        for message in after
        if (
            message.get("role") == "tool"
            and message.get("tool_call_id")
            == "call_001"
        )
    ]

    assert tool_messages, (
        "测试前置预期失败："
        "compression 后应该仍保留 call_001 的 tool result。"
    )

    # ------------------------------------------------------------
    # 核心长期 invariant：
    #
    # 只要 tool(call_001) 被保留，
    # 对应 assistant(tool_calls=call_001)
    # 就必须同时存在。
    #
    # 当前实现会在这里 FAIL。
    # ------------------------------------------------------------
    matching_assistant_exists = any(
        message.get("role") == "assistant"
        and any(
            tool_call.get("id") == "call_001"
            for tool_call in message.get(
                "tool_calls",
                [],
            )
        )
        for message in after
    )
    assert matching_assistant_exists, (
        "\n"
        "Context compression broke Tool Calling protocol.\n\n"
        "tool(call_001) 被保留，"
        "但对应 assistant(tool_calls=call_001) 被裁掉。\n"
        "Tool Calling message block 被拆分。"
    )