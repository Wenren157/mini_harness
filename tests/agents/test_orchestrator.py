"""
测试 Orchestrator 的完整流程：
- Plan → Execute → Aggregate
- Runtime 保持独立
- 返回结构化结果
"""

import pytest
from unittest.mock import AsyncMock, patch
from mini_harness.agents.orchestrator import Orchestrator
from mini_harness.agents.planner import Planner, PlanStep
from mini_harness.agents.executor import Executor


class MockRuntime:
    async def run(self, task_description: str):
        return f"done: {task_description}"


class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_full_goal_execution_returns_structured_result(self):
        runtime = MockRuntime()
        planner = Planner()  # 无 LLM，使用规则
        orchestrator = Orchestrator(runtime, planner)

        result = await orchestrator.run_goal("分析 test.py")
        assert "goal" in result
        assert "plan" in result
        assert "results" in result
        assert "summary" in result
        assert result["goal"] == "分析 test.py"
        assert len(result["plan"]) == 3
        assert len(result["results"]) == 3

    @pytest.mark.asyncio
    async def test_runtime_is_not_polluted_by_agent_logic(self):
        """验证 Runtime 实例没有被 Orchestrator 修改或添加 Agent 相关属性"""
        runtime = MockRuntime()
        planner = Planner()
        orchestrator = Orchestrator(runtime, planner)

        # 保存初始 dir
        original_attrs = set(dir(runtime))
        await orchestrator.run_goal("分析 test.py")
        # 执行后检查 Runtime 实例属性未改变
        assert set(dir(runtime)) == original_attrs

    @pytest.mark.asyncio
    async def test_scope_is_stored(self):
        runtime = MockRuntime()
        planner = Planner()
        scope = {"agent_id": "test", "workspace": "/tmp/test"}
        orchestrator = Orchestrator(runtime, planner, scope=scope)
        assert orchestrator.scope == scope

    def test_orchestrator_does_not_import_execution_components(self):
        """
    Orchestrator不应该依赖Tool/MCP/Sandbox执行层组件。
    检查模块import，而不是检查注释内容。
    """
    import inspect
    import mini_harness.agents.orchestrator as orchestrator_module

    source = inspect.getsource(orchestrator_module)

    assert "from mini_harness.tools" not in source
    assert "ToolRegistry(" not in source
    assert "MCPClient(" not in source
    assert "SandboxExecutor(" not in source