import asyncio
import os
import sys
import tempfile
import pytest
from unittest.mock import AsyncMock, MagicMock

from mini_harness.core.models import EventType
from mini_harness.core.runtime import HarnessRuntime
from mini_harness.infra.config import RuntimeConfig
from mini_harness.infra.tools import ToolRegistry,SandboxExecutor
from mini_harness.core.interfaces import LLMClient


class MockLLM(LLMClient):
    """模拟 LLM，总是返回固定的 tool_calls，用于测试 Runtime→MCP 集成"""
    async def generate(self, messages):
        # 第一次调用返回 write_file，第二次返回 read_file，第三次返回 execute_command
        # 但为了测试简单，我们固定返回一个 read_file 调用，并期望文件已事先存在
        # 或者我们可以设计顺序，但这里我们让测试自己控制
        # 实际上我们会通过不同的测试用例来分别测试三个工具
        return {
            "content": None,
            "tool_calls": [
                {"name": "write_file", "arguments": {"path": "/tmp/test.txt", "content": "Hello MCP"}}
            ]
        }


@pytest.fixture
def config():
    cfg = RuntimeConfig(
        workspace=os.path.abspath("./workspace")
    )
    cfg.enable_mcp = True
    cfg.max_iterations = 5
    cfg.tool_timeout = 5.0
    cfg.enable_concurrent_tools = False
    return cfg


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def tool_registry():
    # 实际上在 MCP 模式下不会使用，但为了构造 Runtime，仍需传入
    # 创建一个 mock sandbox，因为 MCP 模式下不会真正使用
    mock_sandbox = MagicMock(spec=SandboxExecutor)
    return ToolRegistry(sandbox=mock_sandbox)


@pytest.mark.asyncio
async def test_runtime_mcp_integration_write_read(config, mock_llm, tool_registry):
    runtime = HarnessRuntime(config, mock_llm, tool_registry)
    try:
        # 使用 run 方法，让 Agent 循环调用工具
        # 但 mock_llm 只能返回固定的 tool_calls，我们需要让它分步骤返回不同工具
        # 更简单的方法：直接调用 runtime._step 手动控制，或修改 MockLLM 有状态
        # 这里我们使用一个能够按顺序返回不同 tool_calls 的 MockLLM
        # 重新定义一个有状态的 MockLLM
        class SequentialMockLLM(LLMClient):
            def __init__(self):
                self.step = 0
            async def generate(
                     self,
                messages,
                tools=None
            ):
                self.step += 1
                if self.step == 1:
                    # 写入文件
                    return {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "mock_call_1",
                                "name": "write_file", 
                                "arguments": {
                                    "path": "test_mcp.txt", 
                                    "content": "Integration Test"
                                    }
                            }
                        ]
                    }
                elif self.step == 2:
                    # 读取文件
                    return {
                        "content": None,
                        "tool_calls": [
                            {
                                "id":"mock_call_2",
                                "name": "read_file", 
                                "arguments": {
                                    "path": "test_mcp.txt"
                                    }
                            }
                        ]
                    }
                else:
                    # 结束
                    return {
                        "content": "Task completed.",
                        "tool_calls": []
                    }

        runtime.llm = SequentialMockLLM()

        # 执行 run
        result = await runtime.run("Write a file and read it")

        # 验证最终回答包含预期内容
        assert "Task completed" in result["final_answer"]

        # 检查文件是否写入成功
        workspace_file = os.path.join(
            config.workspace,
            "test_mcp.txt"
        )
        with open(workspace_file,"r", encoding="utf-8") as f:
            assert f.read()=="Integration Test"
        

        # 验证事件中有工具调用记录
        events = result["events"]
        tool_call_events = [e for e in events if e.type == EventType.TOOL_CALL_RESULT]
        assert len(tool_call_events) >= 2  # 至少 write 和 read
        # 检查内容
        write_event = next(e for e in tool_call_events if e.data.get("name") == "write_file")
        read_event = next(e for e in tool_call_events if e.data.get("name") == "read_file")
        assert "success" in write_event.data.get("result", "").lower()
        assert "Integration Test" in read_event.data.get("result", "")

    finally:
        await runtime.close()

        # 清理可能残留的文件
        workspace_file = os.path.join(
            config.workspace,
            "test_mcp.txt"
        )

        if os.path.exists(workspace_file):
            os.remove(workspace_file)


@pytest.mark.asyncio
async def test_runtime_mcp_execute_command(config, mock_llm, tool_registry):
    runtime = HarnessRuntime(config, mock_llm, tool_registry)
    try:
        class CmdMockLLM:
            def __init__(self):
                self.called=False

            async def generate(
                self,
                messages,
                tools=None
            ):
                if not self.called:
                    self.called=True
                    return {
                        "content":None,
                        "tool_calls":[
                            {
                                "id":"mock_call_1",
                                "name":"execute_command",
                                "arguments":{
                                "command":"echo hello"
                                }
                            }
                        ]
                    }
                return {
                    "content":"done",
                    "tool_calls":[]
                }
            
        runtime.llm = CmdMockLLM()
        result = await runtime.run("Run echo hello")
        # 验证结果中包含 hello
        # 由于最终 answer 为 None，会设置为 "Task stopped due to max iterations or error."
        # 但我们检查事件
        events = result["events"]
        tool_results = [e for e in events if e.type == EventType.TOOL_CALL_RESULT and e.data.get("name") == "execute_command"]
        assert len(tool_results) == 1
        assert "hello" in tool_results[0].data.get("result", "")
    finally:
        await runtime.close()