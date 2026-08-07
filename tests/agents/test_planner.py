"""
测试 Planner 的核心行为：
- 无 LLM 时能基于规则生成计划
- 计划不绑定 Tool，只输出抽象步骤
- PlanStep 表达任务意图而非执行逻辑
"""

import pytest
from mini_harness.agents.planner import Planner, PlanStep


class TestPlanner:
    @pytest.mark.asyncio
    async def test_rule_based_plan_for_analysis(self):
        """包含“分析”关键词的目标生成三步计划"""
        planner = Planner()  # 无 LLM
        steps = await planner.plan("分析 runtime.py 的结构")
        assert len(steps) == 3
        # 检查步骤顺序与依赖
        assert steps[0].step_id == 1
        assert steps[1].step_id == 2
        assert steps[2].step_id == 3
        assert steps[1].depends_on == [1]
        assert steps[2].depends_on == [2]

    @pytest.mark.asyncio
    async def test_rule_based_plan_for_generation(self):
        """包含“生成”关键词的目标生成两步计划"""
        planner = Planner()
        steps = await planner.plan("生成测试报告")
        assert len(steps) == 2
        assert steps[1].depends_on == [1]

    @pytest.mark.asyncio
    async def test_rule_based_plan_default(self):
        """其他目标生成单步计划"""
        planner = Planner()
        steps = await planner.plan("列出所有文件")
        assert len(steps) == 1
        assert steps[0].description == "列出所有文件"

    def test_planstep_is_intent_not_tool_call(self):
        """PlanStep 只包含自然语言描述，不包含工具名称"""
        step = PlanStep(step_id=1, description="读取 runtime.py 文件")
        assert "read_file" not in step.description.lower()
        assert "tool" not in step.description.lower()

    def test_planner_does_not_import_tool_registry(self):
        """验证 Planner 模块没有导入 ToolRegistry 或 Sandbox"""
        import inspect
        import mini_harness.agents.planner as planner_module

        source = inspect.getsource(planner_module)

        assert "from mini_harness.infra.tool_registry import ToolRegistry" not in source
        assert "ToolRegistry" not in source.split("class Planner")[0]

        assert "MCPClient" not in source.split("class Planner")[0]
        assert "SandboxExecutor" not in source.split("class Planner")[0]