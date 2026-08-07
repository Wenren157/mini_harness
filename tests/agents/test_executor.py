"""
测试 Executor 的行为：
- 通过 Runtime.run() 驱动执行
- 不直接访问 ToolRegistry
- 支持单步 / 多步顺序执行
- 单步失败不阻塞后续步骤
"""

import pytest
from unittest.mock import AsyncMock, patch
from mini_harness.agents.executor import Executor
from mini_harness.agents.planner import PlanStep


class MockRuntime:
    """模拟 HarnessRuntime，仅提供 run 接口"""
    async def run(self, task_description: str):
        return f"executed: {task_description}"


class FailingRuntime:
    """模拟执行失败的 Runtime"""
    async def run(self, task_description: str):
        raise RuntimeError("模拟执行失败")


class TestExecutor:
    @pytest.mark.asyncio
    async def test_execute_single_step_via_runtime_run(self):
        runtime = MockRuntime()
        executor = Executor(runtime)
        step = PlanStep(step_id=1, description="读取文件")
        result = await executor.execute_single_step(step)
        assert "executed: 读取文件" in result

    @pytest.mark.asyncio
    async def test_execute_multiple_steps_in_sequence(self):
        runtime = MockRuntime()
        executor = Executor(runtime)
        steps = [
            PlanStep(step_id=1, description="步骤一"),
            PlanStep(step_id=2, description="步骤二", depends_on=[1]),
        ]
        results = await executor.execute_steps(steps)
        assert len(results) == 2
        assert "executed: 步骤一" in results[1]
        assert "executed: 步骤二" in results[2]

    @pytest.mark.asyncio
    async def test_failure_does_not_block_subsequent_steps(self):
        """
        第一个步骤失败时，Executor应该捕获异常，并继续执行后续步骤。
        """

        runtime = MockRuntime()
        executor = Executor(runtime)

        with patch.object(
            executor.runtime,
            "run",
            side_effect=[
                RuntimeError("失败"),
                "成功"
            ]
        ):
            steps = [
                PlanStep(step_id=1, description="失败步骤"),
                PlanStep(step_id=2, description="成功步骤"),
            ]

            results = await executor.execute_steps(steps)

        assert "error" in results[1]
        assert results[2] == "成功"

    def test_executor_does_not_import_tool_registry(self):
        import inspect
        import mini_harness.agents.executor as executor_module

        source = inspect.getsource(executor_module)

        import_section = source.split("class Executor")[0]

        assert "ToolRegistry" not in import_section
        assert "MCPClient" not in import_section
        assert "SandboxExecutor" not in import_section