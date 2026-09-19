from mini_harness.infra.context import (
    ContextWindow,
    TokenEstimator,
)

def test_hard_trim_does_not_leave_orphan_tool_message():
    """
    P1 invariant:
    Hard Trim 不能拆开：

        assistant(tool_calls)
        +
        matching tool result(s)

    当前 message-level reverse trim 可能出现：

        recent user        -> KEEP
        tool result        -> KEEP
        assistant call     -> TOO LARGE / STOP

    最终留下 orphan tool。

    P1 完成后，assistant + tool 必须作为同一个
    Message Block：

        KEEP whole block
        或
        DROP whole block
    """
    window = ContextWindow(
        max_tokens=1000,
        reserve_tokens=0,
    )
    old_message = {
        "role": "user",
        "content": "old-history",
    }
    assistant_tool_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "dummy_tool",
                    "arguments": "{}",
                },
            }
        ],
    }
    tool_result = {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "tool-result",
    }
    recent_message = {
        "role": "user",
        "content": "recent-user-message",
    }
    window.messages = [
        old_message,
        assistant_tool_call,
        tool_result,
        recent_message,
    ]
    tool_tokens = TokenEstimator.estimate_message(tool_result)
    recent_tokens = TokenEstimator.estimate_message(recent_message)
    assistant_tokens = TokenEstimator.estimate_message(assistant_tool_call)

    # ---------------------------------------------------------
    # 精确制造 message-level trim boundary：
    #
    # recent + tool 能放下，
    # 但 recent + assistant + tool 放不下。
    #
    # 因此旧实现会：
    #
    #     KEEP recent
    #     KEEP tool
    #     DROP assistant
    #
    # 从而制造 orphan tool。
    # ---------------------------------------------------------

    hard_limit = tool_tokens + recent_tokens

    assert hard_limit < (
        assistant_tokens
        + tool_tokens
        + recent_tokens
    )

    window.max_tokens = hard_limit
    window.reserve_tokens = 0

    # 必须让 ContextWindow 进入 hard overflow 分支。
    window.recalculate_total_tokens()
    
    assert window.is_overflow() is True

    trimmed = window.get_messages_for_llm()

    # ---------------------------------------------------------
    # P1 protocol invariant：
    #
    # 任何保留下来的 tool message，
    # 都必须存在对应的 preceding assistant(tool_calls)。
    # ---------------------------------------------------------

    tool_messages = [
        message
        for message in trimmed
        if message.get("role") == "tool"
    ]

    assistant_tool_messages = [
        message
        for message in trimmed
        if (
            message.get("role") == "assistant"
            and message.get("tool_calls")
        )
    ]

    # P1 完成后：
    #
    # 这个 block 因为整体放不下，
    # assistant + tool 应该一起被 DROP。
    assert tool_messages == []
    assert assistant_tool_messages == []

    # 最新普通消息仍应保留。
    assert recent_message in trimmed