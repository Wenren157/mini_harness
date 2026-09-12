"""
ContextWindow Token Accounting Tests

目标：
验证 P0-1:
1. 所有 message 通过 ContextWindow 统计 token
2. assistant tool_calls 有 token accounting
3. tool result 有 token accounting
4. rebuild 后 total_tokens 与 messages 一致
"""
from mini_harness.infra.context import (
    ContextWindow,
    TokenEstimator,
)

def test_normal_message_token_accounting():
    """
    普通消息应该进入 token 统计
    """
    window = ContextWindow()
    before = window.total_tokens
    window.add_message_dict(
        {
            "role": "user",
            "content": "hello world"
        }
    )
    after = window.total_tokens
    assert after > before
    expected = TokenEstimator.estimate_message(
        {
            "role": "user",
            "content": "hello world"
        }
    )
    assert after == expected

def test_assistant_tool_calls_have_token_accounting():
    """
    assistant(tool_calls) 即使 content=None，
    tool_calls 内容也必须参与 token 计算。
    """
    window = ContextWindow()
    tool_call_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_test_001",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path":"demo.py"}'
                }
            }
        ]
    }
    window.add_message_dict(tool_call_message)
    expected = TokenEstimator.estimate_message(tool_call_message)
    assert expected > 0
    assert window.total_tokens == expected

def test_tool_result_has_token_accounting():
    """
    tool message 必须被正常统计。
    """
    window = ContextWindow()
    tool_result = {
        "role": "tool",
        "tool_call_id": "call_test_001",
        "content": (
            "file content "
            * 50
        )
    }

    window.add_message_dict(tool_result)
    expected = TokenEstimator.estimate_message(tool_result)

    assert expected > 0
    assert window.total_tokens == expected

def test_recalculate_total_tokens_matches_messages():
    """
    compression / trim 后重新计算 token，
    必须与当前 messages 完全一致。
    """
    window = ContextWindow()
    messages = [
        {
            "role": "user",
            "content": "inspect project"
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_001",
                    "type": "function",
                    "function": {
                        "name": "list_directory",
                        "arguments": "{}"
                    }
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_001",
            "content": "directory result"
        }
    ]

    for msg in messages:
        window.add_message_dict(msg)

    # 人为制造状态漂移
    window.total_tokens = 0
    recalculated = (window.recalculate_total_tokens())
    expected = sum(
        TokenEstimator.estimate_message(m)
        for m in window.messages
    )
    
    assert recalculated == expected
    assert window.total_tokens == expected