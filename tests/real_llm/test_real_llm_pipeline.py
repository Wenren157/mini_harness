"""
Real LLM End-to-End Pipeline Test

    验证：

    User
    |
    Runtime
    |
    Real LLM
    |
    Tool Calling
    |
    ToolRegistry
    |
    Sandbox
"""
import os
import json
from pathlib import Path

import pytest

from mini_harness.core.runtime import HarnessRuntime
from mini_harness.infra.real_llm_client import RealLLMClient
from mini_harness.infra.tools import create_default_tools
from mini_harness.infra.config import RuntimeConfig


requires_api_key = pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY not configured"
)

@pytest.fixture
def workspace(tmp_path):
    """
    每个测试独立workspace
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return str(workspace)

@pytest.fixture
def runtime(workspace):

    config = RuntimeConfig(
        enable_mcp=False,
        workspace=workspace,
        max_iterations=10,
        tool_timeout=30,
    )
    # create_default_tools已经完成:
    #
    # SandboxExecutor
    # ToolRegistry
    # tool register
    #
    tool_registry = create_default_tools(workspace)
    llm = RealLLMClient(model="deepseek-chat")
    runtime = HarnessRuntime(
        config,
        llm,
        tool_registry
    )
    return runtime

@requires_api_key
@pytest.mark.asyncio
async def test_real_llm_basic_pipeline(runtime):

    result = await runtime.run(
        "你好，请简单介绍一下你自己"
    )
    assert result["final_answer"] is not None

@requires_api_key
@pytest.mark.asyncio
async def test_real_llm_file_task(runtime, workspace):

    task = """
        请帮我完成以下任务：
            1. 创建 hello.txt 文件
            2. 文件内容写入：Hello from Mini Harness
            3. 再读取这个文件确认内容。
    """
    result = await runtime.run(task)
    print(
        "FINAL:",
        result["final_answer"]
    )
    file_path = Path(workspace) / "hello.txt"
    assert file_path.exists()
    content = file_path.read_text(encoding="utf-8")
    assert "Hello from Mini Harness" in content

@requires_api_key
@pytest.mark.asyncio
async def test_real_llm_trace(runtime, workspace):

    result = await runtime.run(
        """
            请执行命令：
            echo Test OK
            然后告诉我结果。
        """
    )
    events = result["events"]
    event_types = [
        e.type.value
        for e in events
    ]

    print(event_types)
    assert "llm_request" in event_types
    assert (
        "tool_call_request" in event_types
        or
        "finish" in event_types
    )