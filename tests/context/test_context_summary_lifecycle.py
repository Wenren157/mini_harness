import pytest
from mini_harness.infra.context import ContextManager

class RecordingSummaryLLM:
    """
    P0-3 专用 deterministic Mock。

    作用：
    1. 记录每一次 summarizer 请求；
    2. 每次返回可预测的 summary；
    3. 不依赖真实 LLM API。
    """
    def __init__(self):
        self.calls = []

    async def generate(self, messages):
        self.calls.append(messages)
        call_number = len(self.calls)
        return {
            "content": f"summary-{call_number}",
            "tool_calls": [],
        }

def _get_history_summaries(manager: ContextManager):
    """
    返回当前 Context 中所有历史摘要 message。
    """
    return [
        message
        for message in manager.window.messages
        if (
            message.get("role") == "system"
            and isinstance(message.get("content"), str)
            and message["content"].startswith("[历史摘要]")
        )
    ]

@pytest.mark.asyncio
async def test_second_compression_replaces_existing_history_summary():
    """
    P0-3 invariant:
    Context 中最多只能存在一个逻辑历史摘要。
    第二次 compression 应该：
        old summary + newly evicted raw history -> new summary
    而不是：
        old summary
        +
        new summary
    """
    llm = RecordingSummaryLLM()
    manager = ContextManager(
        llm_client=llm,
        max_tokens=8000,
    )
    manager.add_system_message("You are a helpful assistant.")

    # ---------------------------------------------------------
    # 第一次 compression
    #
    # 5 个普通 block：
    #
    # old-1
    # old-2
    # recent-1
    # recent-2
    # recent-3
    #
    # P0-2 规则：
    # old-1 / old-2 -> summary
    # recent-*      -> raw
    # ---------------------------------------------------------
    manager.add_user_message("old-1")
    manager.add_user_message("old-2")
    manager.add_user_message("recent-1")
    manager.add_user_message("recent-2")
    manager.add_user_message("recent-3")
    await manager._compress_background()

    first_summaries = _get_history_summaries(manager)

    assert len(first_summaries) == 1
    assert first_summaries[0]["content"] == ("[历史摘要] summary-1")

    # ---------------------------------------------------------
    # 增加两条消息，使第二轮再次拥有 5 个 raw block。
    #
    # 当前 raw：
    #
    # recent-1
    # recent-2
    # recent-3
    # new-1
    # new-2
    #
    # 第二次 compression 会淘汰：
    #
    # recent-1
    # recent-2
    # ---------------------------------------------------------

    manager.add_user_message("new-1")
    manager.add_user_message("new-2")

    await manager._compress_background()

    second_summaries = _get_history_summaries(manager)

    # ---------------------------------------------------------
    # P0-3 核心 invariant：
    #
    # 旧 summary 必须被新 summary 替换，
    # Context 中不能不断累积多个历史摘要。
    # ---------------------------------------------------------
    assert len(second_summaries) == 1
    assert second_summaries[0]["content"] == ("[历史摘要] summary-2")

@pytest.mark.asyncio
async def test_second_compression_rolls_old_summary_into_new_summary_input():
    """
    P0-3 invariant:
    第二次 summarization 的输入必须包含：
        old summary
        +
        newly evicted raw history
    防止 rolling summary 更新时丢失更早历史。
    """
    llm = RecordingSummaryLLM()
    manager = ContextManager(
        llm_client=llm,
        max_tokens=8000,
    )
    manager.add_system_message("You are a helpful assistant.")

    manager.add_user_message("old-1")
    manager.add_user_message("old-2")
    manager.add_user_message("recent-1")
    manager.add_user_message("recent-2")
    manager.add_user_message("recent-3")
    await manager._compress_background()

    assert len(llm.calls) == 1
    manager.add_user_message("new-1")
    manager.add_user_message("new-2")

    await manager._compress_background()
    assert len(llm.calls) == 2

    second_summary_request = llm.calls[1]

    # summarizer 的 user message
    second_summary_source = (second_summary_request[1]["content"])

    # ---------------------------------------------------------
    # 旧 rolling summary 必须参与下一次 summary。
    # ---------------------------------------------------------
    assert "[历史摘要] summary-1" in second_summary_source

    # ---------------------------------------------------------
    # 本轮真正被淘汰的 raw history 也必须进入 summary。
    # ---------------------------------------------------------
    assert "recent-1" in second_summary_source
    assert "recent-2" in second_summary_source

    # ---------------------------------------------------------
    # 最近 3 个 block 仍然必须保持 raw，
    # 不能重复塞进 summary。
    # ---------------------------------------------------------
    assert "recent-3" not in second_summary_source
    assert "new-1" not in second_summary_source
    assert "new-2" not in second_summary_source

@pytest.mark.asyncio
async def test_rolling_summary_preserves_pinned_system_message():
    """
    P0-3 invariant:
    pinned system instruction 与 historical summary
    是两种不同生命周期的数据。

    rolling summary 更新不能删除、覆盖或重复
    pinned system instruction。
    """

    llm = RecordingSummaryLLM()
    manager = ContextManager(
        llm_client=llm,
        max_tokens=8000,
    )
    pinned_system = "You are a helpful assistant."
    manager.add_system_message(pinned_system)
    manager.add_user_message("old-1")
    manager.add_user_message("old-2")
    manager.add_user_message("recent-1")
    manager.add_user_message("recent-2")
    manager.add_user_message("recent-3")

    await manager._compress_background()
    manager.add_user_message("new-1")
    manager.add_user_message("new-2")

    await manager._compress_background()
    pinned_messages = [
        message
        for message in manager.window.messages
        if (
            message.get("role") == "system"
            and message.get("content") == pinned_system
        )
    ]

    history_summaries = _get_history_summaries(manager)
    assert len(pinned_messages) == 1

    # P0-3 完成后：
    # pinned system + one rolling summary
    assert len(history_summaries) == 1