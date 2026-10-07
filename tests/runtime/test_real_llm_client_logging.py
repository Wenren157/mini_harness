import json
import logging
from types import SimpleNamespace

import pytest

from mini_harness.infra.real_llm_client import RealLLMClient


class SuccessfulCompletions:
    def __init__(
        self,
        response_content,
        tool_argument_path,
    ):
        self.response_content = response_content
        self.tool_argument_path = tool_argument_path
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs

        tool_call = SimpleNamespace(
            id="call_read_1",
            function=SimpleNamespace(
                name="read_file",
                arguments=json.dumps(
                    {
                        "path": self.tool_argument_path,
                    }
                ),
            ),
        )

        message = SimpleNamespace(
            content=self.response_content,
            tool_calls=[tool_call],
        )

        choice = SimpleNamespace(
            message=message,
            finish_reason="tool_calls",
        )

        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
        )

        return SimpleNamespace(
            choices=[choice],
            usage=usage,
        )


class FailingCompletions:
    async def create(self, **kwargs):
        raise RuntimeError(
            "planned provider failure"
        )


def build_client(completions):
    """构造不发起网络请求的 RealLLMClient。"""
    client = object.__new__(RealLLMClient)
    client.model = "deepseek-chat"
    client.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=completions
        )
    )
    return client


@pytest.mark.asyncio
async def test_real_llm_logs_metadata_without_exposing_payload(
    capsys,
    caplog,
):
    """LLM 日志应保留结构信息，但不记录请求和响应原文。"""
    sensitive_prompt = "SECRET_USER_PROMPT"
    sensitive_tool_description = "SECRET_TOOL_DESCRIPTION"
    sensitive_response = "SECRET_MODEL_RESPONSE"
    sensitive_path = "private/source.py"

    completions = SuccessfulCompletions(
        response_content=sensitive_response,
        tool_argument_path=sensitive_path,
    )
    client = build_client(completions)

    messages = [
        {
            "role": "user",
            "content": sensitive_prompt,
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": sensitive_tool_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                        }
                    },
                },
            },
        }
    ]

    with caplog.at_level(
        logging.DEBUG,
        logger="mini_harness.infra.real_llm_client",
    ):
        result = await client.generate(
            messages=messages,
            tools=tools,
        )

    captured = capsys.readouterr()

    assert result["content"] == sensitive_response
    assert result["finish_reason"] == "tool_calls"
    assert result["tool_calls"] == [
        {
            "id": "call_read_1",
            "name": "read_file",
            "arguments": {
                "path": sensitive_path,
            },
        }
    ]

    assert captured.out == ""
    assert captured.err == ""

    assert "model=deepseek-chat" in caplog.text
    assert "message_count=1" in caplog.text
    assert "tool_count=1" in caplog.text
    assert "finish_reason=tool_calls" in caplog.text
    assert "tool_call_count=1" in caplog.text

    assert sensitive_prompt not in caplog.text
    assert sensitive_tool_description not in caplog.text
    assert sensitive_response not in caplog.text
    assert sensitive_path not in caplog.text


@pytest.mark.asyncio
async def test_real_llm_logs_failure_without_printing(
    capsys,
    caplog,
):
    """Provider 异常应进入 ERROR 日志，不应直接打印。"""
    client = build_client(
        FailingCompletions()
    )

    with caplog.at_level(
        logging.ERROR,
        logger="mini_harness.infra.real_llm_client",
    ):
        with pytest.raises(
            RuntimeError,
            match="planned provider failure",
        ):
            await client.generate(
                messages=[
                    {
                        "role": "user",
                        "content": "trigger failure",
                    }
                ],
                tools=None,
            )

    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == ""

    assert "LLM request failed" in caplog.text
    assert "planned provider failure" in caplog.text