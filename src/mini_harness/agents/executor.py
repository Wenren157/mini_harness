"""
Executor Agent：顺序执行 Planner 产生的计划步骤。

设计决策：
- Executor 为什么不能直接调用 Tool？
  为了统一管理 Context、MCP、Trace 与 Error Handling。
  所有工具调用必须经过 Runtime，避免执行路径分散导致上下文污染或 trace 断裂。

- Executor 与 Planner 为什么分离？
  规划与执行隔离，单一职责。Planner 只输出抽象步骤，Executor 只负责逐步驱动 Runtime，
  这样可以在不修改 Executor 的前提下替换 Planner 策略（如 LLM 规划、动态重规划）。

- Executor 不负责选择工具，工具选择完全由 Runtime 根据 step.description 自动完成，
  因此 Executor 与具体 Tool/MCP 实现解耦。
"""

from typing import List, Dict, Any

from mini_harness.core.runtime import HarnessRuntime
from mini_harness.agents.planner import PlanStep
from mini_harness.agents.scope import AgentScope


class Executor:
    """
    Executor Agent：按顺序执行 Planner 产生的步骤。
    通过 Runtime 调用工具，不直接访问 ToolRegistry。

    注意：
    - Executor 不负责任务规划。
    - Executor 不负责决定调用哪个 Tool。
    - Executor 只负责执行 Planner 输出的 PlanStep。

    所有工具调用必须经过 Runtime，
    保证：
    - MCP 链路
    - Context 记录
    - EventBus 事件
    - Trace 追踪
    保持一致。
    """

    def __init__(
            self, 
            runtime: HarnessRuntime,
            scope: AgentScope
    ):
        self.runtime = runtime
        self.scope = scope

    async def execute_steps(self, steps: List[PlanStep]) -> Dict[int, Any]:
        """
        顺序执行步骤，返回每个步骤的结果。

        Args:
            steps: Planner 产生的待执行 PlanStep 列表。

        Returns:
            {step_id: result} 的执行结果字典。

        Example:
            {
                1: "runtime.py 读取完成",
                2: "代码结构分析完成"
            }
        """
        results = {}
        for step in steps:
            try:
                result = await self.execute_single_step(step)
                results[step.step_id] = result
            except Exception as e:
                # 单个步骤失败不影响后续步骤执行，记录错误信息
                results[step.step_id] = {
                    "error": str(e),
                    "step": step.description,
                }
        return results

    async def execute_single_step(self, step: PlanStep) -> Any:
        """
        执行单个步骤，供 Orchestrator 细粒度调用。

        调用链路：

            Executor
                ↓
            HarnessRuntime.run()
                ↓
            MCP / ToolRegistry

        禁止：

            Executor → ToolRegistry
            Executor → MCPClient

        Executor只负责驱动 Runtime，
        不参与具体执行逻辑。
        """

        return await self.runtime.run(
            step.description
        )