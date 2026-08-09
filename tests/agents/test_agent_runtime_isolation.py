"""
Agent Runtime Isolation Tests

Day6 Final Verification:

1. Runtime 保存自己的 AgentScope
2. Agent Workspace 隔离
3. Agent Context 隔离

验证目标：
Agent A:

    AgentScope(A)
          |
    Runtime(A)
          |
    Workspace(A)
          |
    Context(A)

Agent B:

    AgentScope(B)
          |
    Runtime(B)
          |
    Workspace(B)
          |
    Context(B)

A / B 不产生状态污染。
"""

import pytest
from mini_harness.agents.scope import AgentScope
from mini_harness.core.runtime import HarnessRuntime

class DummyLLM:
    async def generate(self, *args, **kwargs):
        return {
            "content": "done",
            "tool_calls": []
        }

class DummyToolRegistry:
    def get_schema(self):
        return []
    
    async def execute_with_retry(self, *args, **kwargs):
        return "ok"

class DummyContextManager:
    """
    简化 ContextManager。
    只验证 Runtime 是否拥有独立 context。
    """
    def __init__(self):
        self.messages = []

    def add_user_message(self, message):
        self.messages.append(message)

class DummyConfig:
    event_bus_maxlen = 100
    max_context_tokens = 1000
    enable_mcp = False
    workspace = None

def create_runtime(scope):
    runtime = HarnessRuntime(
        config=DummyConfig(),
        llm_client=DummyLLM(),
        tool_registry=DummyToolRegistry(),
        context_manager=DummyContextManager(),
        scope=scope,
    )
    return runtime

# ==========================================================
# Test 1
# Runtime 保存自己的 Scope
# ==========================================================
def test_runtime_keeps_agent_scope(tmp_path):

    scope_a = AgentScope(
        agent_id="agent_a",
        workspace=tmp_path / "agent_a"
    )
    scope_b = AgentScope(
        agent_id="agent_b",
        workspace=tmp_path / "agent_b"
    )

    runtime_a = create_runtime(scope_a)
    runtime_b = create_runtime(scope_b)

    # Runtime 保存自己的 Scope
    assert runtime_a.scope == scope_a
    assert runtime_b.scope == scope_b

    # Agent identity 不混淆
    assert runtime_a.agent_id == "agent_a"
    assert runtime_b.agent_id == "agent_b"
    assert runtime_a.agent_id != runtime_b.agent_id

# ==========================================================
# Test 2
# Workspace 隔离
# ==========================================================
def test_runtime_workspace_isolation(tmp_path):

    workspace_a = tmp_path / "agent_a"
    workspace_b = tmp_path / "agent_b"
    scope_a = AgentScope(
        agent_id="agent_a",
        workspace=workspace_a
    )
    scope_b = AgentScope(
        agent_id="agent_b",
        workspace=workspace_b
    )
    runtime_a = create_runtime(scope_a)
    runtime_b = create_runtime(scope_b)

    assert str(runtime_a.workspace) == str(workspace_a)
    assert str(runtime_b.workspace) == str(workspace_b)
    assert runtime_a.workspace != runtime_b.workspace

    # 模拟 Agent A 创建文件
    file_a = workspace_a / "a.txt"
    file_a.write_text(
        "agent_a"
    )
    assert file_a.exists()
    assert not (workspace_b / "a.txt").exists()

    # Agent B workspace 不应该出现
    assert not (workspace_b / "a.txt").exists()

# ==========================================================
# Test 3
# Context 隔离
# ==========================================================
def test_runtime_context_isolation(tmp_path):

    scope_a = AgentScope(
        agent_id="agent_a",
        workspace=tmp_path / "agent_a"
    )
    scope_b = AgentScope(
        agent_id="agent_b",
        workspace=tmp_path / "agent_b"
    )
    runtime_a = create_runtime(scope_a)
    runtime_b = create_runtime(scope_b)

    # Agent A 写入自己的 Context
    runtime_a.context.add_user_message(
        "Agent A analysis runtime.py"
    )

    # Context对象必须不同
    assert runtime_a.context is not runtime_b.context

    # A 能看到自己的消息
    assert (
        "Agent A analysis runtime.py"
        in runtime_a.context.messages
    )

    # B 不应该看到 A 的消息
    assert (
        "Agent A analysis runtime.py"
        not in runtime_b.context.messages
    )