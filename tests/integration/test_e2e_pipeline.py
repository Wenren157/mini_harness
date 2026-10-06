import asyncio
import os
import json
import pytest

from mini_harness.core.runtime import (
    HarnessRuntime,
    RuntimeConfig
)
from mini_harness.infra.tools import (
    create_default_tools,
)
from mini_harness.core.models import EventType

class E2EMockLLM:
    """
    模拟真实Agent行为：
    第一次:请求调用工具
    第二次:根据tool结果结束

    """

    def __init__(self):
        self.calls = 0

    async def generate(
            self, 
            messages,
            tools=None
    ):

        self.calls += 1

        # 第一次进入Agent Loop
        if self.calls == 1:

            return {
                "content": None,
                "tool_calls": [
                    {
                        "name": "read_file",
                        "arguments": {
                            "path": "a.txt"
                        },
                        "id": "call_read_1"
                    }
                ]
            }

        # 后续返回最终结果
        return {
            "content": "文件处理完成",
            "tool_calls": []
        }

@pytest.mark.asyncio
async def test_e2e_runtime_tool_context_trace(tmp_path):

    """
    End-to-End Pipeline:

    Runtime
        |
        Tool
        |
        Context
        |
        Compression
        |
        EventBus
        |
        Trace
    """
    # ============================
    # 1. 创建真实组件
    # ============================

    config = RuntimeConfig(
        # 使用合法的 Soft/Hard Threshold 区间触发压缩
        max_context_tokens=2500,
        max_iterations=5,
        event_bus_maxlen=200,
        tool_timeout=3.0
    )
    llm = E2EMockLLM()

    test_file = tmp_path / "a.txt"
    test_file.write_text(
        "hello from e2e test",
        encoding="utf-8",
    )

    tools = create_default_tools(
        workspace=str(tmp_path)
    )
    runtime = HarnessRuntime(
        config=config,
        llm_client=llm,
        tool_registry=tools
    )

    # ============================
    # 2. 第一次真实Agent运行
    # ============================
    result = await runtime.run("请读取a.txt文件")
    assert result["final_answer"] == "文件处理完成"

    # 保存第一次完整链路事件
    tool_events = list(runtime.event_bus)

    # ============================
    # 3. 制造上下文压力
    #    触发compression
    # ============================
    for i in range(8):
        await runtime.run(
            f"""
            additional message {i}
            fill context window
            {"hello " * 50}
            """
        )
    # 等待后台压缩任务
    await asyncio.sleep(1)

    # ============================
    # 4. 验证 Tool 链路
    # ============================
    events = tool_events
    event_types = [
        e.type
        for e in events
    ]
    assert (
        EventType.TOOL_CALL_REQUEST
        in event_types
    ), "没有Tool Call事件"
    assert (
        EventType.TOOL_CALL_RESULT
        in event_types
    ), "没有Tool Result事件"

    tool_result_events = [
        event
        for event in tool_events
        if event.type == EventType.TOOL_CALL_RESULT
    ]

    assert len(tool_result_events) == 1

    tool_result = tool_result_events[0].data

    assert tool_result["name"] == "read_file"
    assert "hello from e2e test" in str(tool_result["result"])
    assert "未注册" not in str(tool_result["result"])
    assert "KeyError" not in str(tool_result["result"])

    # ============================
    # 5. 验证 Compression事件
    # ============================

    traces = os.listdir("traces")
    trace_files = [
        f
        for f in traces
        if f.startswith("trace_")
    ]

    assert len(trace_files) > 0

    latest = max(
        trace_files,
        key=lambda x:
        os.path.getmtime(
            os.path.join(
                "traces",
                x
            )
        )
    )

    with open(
        os.path.join(
            "traces",
            latest
        ),
        encoding="utf-8"
    ) as f:
        trace_events = json.load(f)

    actions = [
        e["data"]["action"]
        for e in trace_events
        if isinstance(e.get("data"), dict)
        and "action" in e["data"]
    ]

    assert (
        "compress_start"
        in actions
    ), "缺少compression start事件"

    assert (
        "compress_done"
        in actions
    ), "缺少compression done事件"

    # ============================
    # 6. 最终打印
    # ============================
    print("\n===== E2E Pipeline Success =====")
    print(
        "Events:",
        [
            e.type.value
            for e in events
        ]
    )
    print("Trace:", latest)