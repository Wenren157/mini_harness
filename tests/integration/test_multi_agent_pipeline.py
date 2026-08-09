"""
Day6 Multi-Agent Integration Test

验证：

User
 |
Orchestrator
 |
Planner
 |
Executor
 |
HarnessRuntime
 |
LLM
 |
ToolRegistry
 |
Sandbox
 |
Result
"""

import pytest
from pathlib import Path

from mini_harness.core.runtime import HarnessRuntime
from mini_harness.agents.planner import Planner
from mini_harness.agents.orchestrator import Orchestrator

from mini_harness.infra.config import RuntimeConfig
from mini_harness.infra.tools import create_default_tools


class MockLLMClient:
    """
    模拟LLM Function Calling行为

    第一轮：
        返回write_file
    第二轮：
        返回read_file
    第三轮：
        返回final answer
    """

    def __init__(self):
        self.calls = 0

    async def generate(self, messages, tools=None):
        
        self.calls += 1
        if self.calls == 1:
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_write_001",
                        "name": "write_file",
                        "arguments": {
                            "path": "hello.txt",
                            "content": "Hello Multi Agent"
                        }
                    }
                ]
            }
        if self.calls == 2:
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_read_001",
                        "name": "read_file",
                        "arguments": {
                            "path": "hello.txt"
                        }
                    }
                ]
            }
        return {
            "content": "文件创建并读取成功",
            "tool_calls": []
        }

@pytest.fixture
def runtime(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = RuntimeConfig(
        enable_mcp=False,
        workspace=str(workspace),
        max_iterations=10
    )

    tools = create_default_tools(str(workspace))
    llm = MockLLMClient()
    return HarnessRuntime(
        config,
        llm,
        tools
    ), workspace

@pytest.mark.asyncio
async def test_multi_agent_pipeline(runtime):
    runtime, workspace = runtime
    planner = Planner()
    orchestrator = Orchestrator(
        runtime,
        planner,
        scope={
            "agent_id": "integration-test",
            "workspace": str(workspace)
        }
    )
    result = await orchestrator.run_goal(
        """
        创建 hello.txt 文件，
        写入 Hello Multi Agent，
        然后读取确认
        """
    )
    
    # Orchestrator层验证
    assert result["goal"]
    assert len(result["plan"]) >= 1
    assert result["results"]

    # Runtime最终结果
    assert "文件创建并读取成功" in result["summary"]

    # Tool真实执行验证
    file_path = (
        workspace /
        "hello.txt"
    )
    assert file_path.exists()
    content = file_path.read_text(encoding="utf-8")
    assert content == "Hello Multi Agent"

    # Trace验证
    assert result["results"]