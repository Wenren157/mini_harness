"""
Orchestrator：协调 Planner 和 Executor，管理 Multi-Agent 流程。

设计决策：
- 为什么 Orchestrator 独立于 Runtime？
  Agent 编排层与工具执行层分离，Runtime 保持冻结，Multi-Agent 逻辑通过 Orchestrator 注入，
  保证核心循环稳定，未来可平滑扩展为多 Agent 协作系统。

- 为什么通过 Executor 执行而非直接调用 Runtime？
  Executor 提供统一的步骤执行接口（顺序、容错），Orchestrator 仅负责流程编排，
  遵循单一职责原则，避免 Orchestrator 耦合执行细节。

- 为什么 Agent 需要 Scope？
  支持资源隔离，未来并发 Agent 可通过 scope 分配独立的 workspace、memory namespace 等，
  避免全局污染，这是 Multi-Agent 安全运行的基础。
"""

from typing import Dict, Any, List, Optional
from mini_harness.core.runtime import HarnessRuntime
from mini_harness.agents.planner import Planner, PlanStep
from mini_harness.agents.executor import Executor
from mini_harness.agents.scope import AgentScope


class Orchestrator:
    """
    Orchestrator：协调 Planner 和 Executor，管理 Multi-Agent 流程。

    隔离模式：不污染 Runtime 核心循环。

    职责：
        - 流程控制：Plan → Execute → Aggregate
        - Agent Scope 管理

    禁止：
        - 直接调用 Tool
        - 直接操作 Runtime 内部调度
        - 直接访问 Sandbox
    """

    def __init__(
        self,
        runtime: HarnessRuntime,
        planner: Planner,
        scope: Optional[AgentScope] = None
    ):
        self.runtime = runtime
        self.planner = planner

        # Agent 独立资源范围，未来用于 workspace / context / memory 隔离
        self.scope = scope 
        
        self.executor = Executor(
            runtime,
            scope
        )

    async def run_goal(self, goal: str) -> Dict[str, Any]:
        """
        执行一个完整目标：Plan → Execute → Aggregate。

        Args:
            goal: 用户目标描述

        Returns:
            {
                "goal": str,
                "plan": List[PlanStep],      # 原始 PlanStep 对象列表
                "results": Dict[int, Any],   # {step_id: result}
                "summary": str,
            }
        """
        # 1. 获取当前 Agent 能力描述（不直接访问 ToolRegistry）
        #    当前实现传递空列表，Planner 使用规则拆解时不依赖具体能力描述
        available_capabilities: List[str] = []

        # 2. 调用 Planner 生成计划
        steps: List[PlanStep] = await self.planner.plan(
            goal=goal,
            available_capabilities=available_capabilities,
        )

        # 3. 调用 Executor 顺序执行步骤
        results: Dict[int, Any] = await self.executor.execute_steps(steps)

        # 4. 汇总结果生成摘要（简单字符串拼接）
        summary_parts = []
        for step_id, result in results.items():
            if isinstance(result, dict) and "error" in result:
                summary_parts.append(f"步骤 {step_id} 失败: {result['error']}")
            else:
                summary_parts.append(f"步骤 {step_id} 完成: {str(result)[:100]}")
        summary = "; ".join(summary_parts) if summary_parts else "无结果"

        # 5. 返回结构化结果
        return {
            "goal": goal,
            "plan": steps,
            "results": results,
            "summary": summary,
        }

    # TODO: 支持 Agent 间消息传递（MessageBus）
    # TODO: 支持 DAG 依赖调度（拓扑排序执行）
    # TODO: 支持多 Agent 辩论/投票机制