"""
Planner Agent：将用户目标拆解为可执行步骤序列。

设计决策：
- 为什么 Planner 不绑定具体 Tool？
  规划与工具选择解耦，工具绑定延迟到 Runtime，便于更换底层工具实现而无需修改 Planner。

- 为什么 Planner 与 Executor 分离？
  规划与执行职责隔离。规划只关注“做什么”，执行关注“怎么做”。
  二者独立演化，未来可替换为 LLM 动态规划或并行执行器，互不影响。

- LLM 辅助拆解为什么是可选的？
  保证离线可用性与可控性，规则兜底确保基本功能不依赖外部模型。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any
import json
import re

from mini_harness.infra.llm_client import LLMClient


@dataclass
class PlanStep:
    """
    规划步骤。

    注意：
    Planner 只负责任务拆解，不负责决定具体 Tool 调用。
    Tool 选择和执行由 Runtime 负责。
    """
    step_id: int
    description: str
    depends_on: List[int] = field(default_factory=list)


class Planner:
    """
    Planner Agent：基于规则 + LLM（可选）进行任务拆解。

    职责：
        - 分析用户目标
        - 生成有序的抽象步骤列表
        - 支持基于反馈的重新规划（预留扩展点）

    禁止：
        - 直接输出 tool_name 或 tool_args
        - 访问 ToolRegistry、MCP 等执行层组件
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client

    async def plan(
        self,
        goal: str,
        available_capabilities: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> List[PlanStep]:
        """
        根据目标和可用能力生成执行计划。

        Args:
            goal: 用户目标，如 "分析 mini_harness/runtime.py 的代码结构"
            available_capabilities: 可用能力描述列表（仅用于辅助规划）
            context: 可选上下文信息

        Returns:
            规划步骤列表，按执行顺序排列
        """
        # 优先尝试 LLM 拆解（如果配置了 LLMClient）
        if self.llm_client:
            try:
                return await self._plan_with_llm(goal, available_capabilities, context)
            except Exception:
                # LLM 失败时回退到规则拆解
                pass
        # 规则拆解兜底
        return self._plan_with_rules(goal, available_capabilities, context)

    # --------------------------------------------------------------
    # 规则拆解（确定性）
    # --------------------------------------------------------------
    def _plan_with_rules(
        self,
        goal: str,
        capabilities: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> List[PlanStep]:
        """
        基于关键词的简单规则拆解。
        当前规则集可视为原型，未来可由领域专家或更复杂的策略替换。
        """
        goal_lower = goal.lower()

        # 如果目标包含 "分析" 或 "检查" 等动词，拆解为 读取→分析→总结 三部曲
        if any(kw in goal_lower for kw in ["分析", "检查", "审查", "审计"]):
            return [
                PlanStep(step_id=1, description="读取并加载目标文件或数据"),
                PlanStep(step_id=2, description="分析结构或内容", depends_on=[1]),
                PlanStep(step_id=3, description="整理分析结果", depends_on=[2]),
            ]

        # 如果目标包含 "生成" 或 "创建"，拆解为 收集信息→生成
        if any(kw in goal_lower for kw in ["生成", "创建", "构建"]):
            return [
                PlanStep(step_id=1, description="收集所需的输入信息"),
                PlanStep(step_id=2, description="执行生成任务", depends_on=[1]),
            ]

        # 默认：单一任务步骤，直接执行
        return [PlanStep(step_id=1, description=goal)]

    # --------------------------------------------------------------
    # LLM 辅助拆解（可选）
    # --------------------------------------------------------------
    async def _plan_with_llm(
        self,
        goal: str,
        capabilities: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> List[PlanStep]:
        """
        使用 LLM 将目标拆解为抽象步骤。
        提示词强调只输出自然语言步骤，不包含工具名称。
        """
        # 构建提示词
        prompt = self._build_planning_prompt(goal, capabilities, context)
        # 调用 LLM（假设 LLMClient 提供异步 generate 方法）
        response = await self.llm_client.generate(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )

        # 尝试从 LLM 响应中提取步骤列表（JSON 格式）
        content = response.get("content", "")
        steps_data = self._extract_steps_from_response(content)
        if not steps_data:
            # 如果解析失败，回退到规则拆解
            return self._plan_with_rules(goal, capabilities, context)

        # 转换为 PlanStep 对象
        steps = []
        for item in steps_data:
            step = PlanStep(
                step_id=item.get("step_id", len(steps) + 1),
                description=item.get("description", ""),
                depends_on=item.get("depends_on", []),
            )
            steps.append(step)
        return steps

    def _build_planning_prompt(
        self,
        goal: str,
        capabilities: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> str:
        """
        构造 LLM 规划提示词，确保输出格式符合预期。
        """
        caps_text = ", ".join(capabilities) if capabilities else "通用任务处理能力"
        ctx_text = f"\n上下文信息：{context}" if context else ""

        prompt = f"""你是一个任务规划器。请将以下目标拆解成若干抽象步骤。
目标：{goal}
可用能力：{caps_text}{ctx_text}

要求：
- 步骤用自然语言描述，不要提及任何具体的工具名称、函数名或技术实现细节。
- 步骤应具有逻辑顺序，如果存在依赖关系，请用 depends_on 字段表示（依赖步骤的编号列表）。
- 输出一个 JSON 数组，每个元素包含：
  - "step_id": 整数，从 1 开始递增
  - "description": 字符串，步骤描述
  - "depends_on": 整数数组，依赖的步骤 ID 列表，无依赖时为空数组

只输出 JSON 数组，不要包含任何其他文本。
"""
        return prompt

    def _extract_steps_from_response(self, response: str) -> Optional[List[dict]]:
        """
        从 LLM 原始响应中提取步骤 JSON。
        使用简单正则匹配，避免复杂依赖。
        """
        # 尝试直接解析整个响应
        try:
            return json.loads(response)
        except Exception:
            pass

        # 尝试提取 JSON 数组片段
        pattern = r'\[[\s\S]*\]'
        match = re.search(pattern, response)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

        return None

    # --------------------------------------------------------------
    # 动态重规划扩展点（预留）
    # --------------------------------------------------------------
    def refine_plan(
        self,
        steps: List[PlanStep],
        feedback: str,
    ) -> List[PlanStep]:
        """
        根据执行反馈动态调整计划（预留扩展点）。

        TODO: 未来可接入 LLM 重新规划，或根据失败步骤重新生成后续步骤。
        """
        # 当前版本直接返回原计划，不做调整
        return steps