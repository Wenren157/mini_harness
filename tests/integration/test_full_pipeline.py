import asyncio
import pytest
from mini_harness.core.runtime import HarnessRuntime, RuntimeConfig
from mini_harness.infra.tools import SandboxExecutor, ToolRegistry
from mini_harness.infra.llm_client import OpenAILLMClient  # 或 Mock

class MockLLMClient:
    async def generate(
        self,
        messages,
        tools=None,
    ):
        return {
            "content": "Mock response",
            "tool_calls": [],
        }

@pytest.mark.asyncio
async def test_compression_and_trace():
    config = RuntimeConfig(
        max_iterations=5, 
        max_context_tokens=2000,    # 触发压缩
        event_bus_maxlen=10         # 方便测试裁剪
    )  # 小窗口触发压缩

    llm = MockLLMClient()

    sandbox = SandboxExecutor()
    tools = ToolRegistry(
        sandbox=sandbox
    )

    runtime = HarnessRuntime(config, llm, tools)

    print(
        "runtime event_bus",
        runtime.event_bus
    )

    print(
        "context event_bus",
        runtime.context.event_bus
    )

    print(
        runtime.event_bus is runtime.context.event_bus
    )

    # 发送多条消息，使 token 超限
    for i in range(15):
        result = await runtime.run(
            f"Message {i} with content to fill tokens " * 20
        )

        assert result["final_answer"] == "Mock response"

    # 等待后台压缩完成
    await asyncio.sleep(1)

    # 验证 traces 目录存在且有文件
    import os
    trace_files = os.listdir("traces")
    assert len(trace_files) > 0

    # 验证 event_bus 被裁剪（如果设置了 maxlen）
    if config.event_bus_maxlen:
        assert len(runtime.event_bus) <= config.event_bus_maxlen, "event_bus not truncated"

    # 读取最新的 trace 文件，检查压缩事件
    import json
    trace_files = [
        f for f in os.listdir("traces")
        if f.startswith("trace_")
    ]

    assert len(trace_files) > 0, "No trace file generated"

    found_compress_start = False
    found_compress_done = False


    for trace_file in trace_files:

        trace_path = os.path.join(
            "traces",
            trace_file
        )

        with open(
            trace_path,
            "r",
            encoding="utf-8"
        ) as f:
            events = json.load(f)


        actions = [
            e["data"]["action"]
            for e in events
            if isinstance(e.get("data"), dict)
            and "action" in e["data"]
        ]


        if "compress_start" in actions:
            found_compress_start = True

        if "compress_done" in actions:
            found_compress_done = True


    assert found_compress_start, \
        "compress_start event not found in any trace"

    assert found_compress_done, \
        "compress_done event not found in any trace"