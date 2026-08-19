#!/usr/bin/env python3
"""
Day7 最终 Demo：Mini Harness 自我架构审计（只读 + 可视化执行链路）

运行方式：
    python demo/self_architecture_audit.py

依赖：
    - 环境变量 DEEPSEEK_API_KEY 已设置
    - 项目根目录在 PYTHONPATH 中
"""

import asyncio
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, Any, List, Optional

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

from mini_harness.core.runtime import HarnessRuntime
from mini_harness.core.models import EventType, Event
from mini_harness.infra.real_llm_client import RealLLMClient
from mini_harness.infra.tools import ToolRegistry, SandboxExecutor
from mini_harness.infra.context import ContextManager
from mini_harness.infra.config import RuntimeConfig
from mini_harness.agents.planner import Planner, PlanStep
from mini_harness.agents.orchestrator import Orchestrator
from mini_harness.agents.executor import Executor
from mini_harness.agents.scope import AgentScope


# ============================================================
# 只读工具定义
# ============================================================

async def list_directory(sandbox: SandboxExecutor, path: str = ".") -> Dict[str, Any]:
    """
    列出沙箱内指定目录的树形结构（只读）。
    返回 JSON 字符串，包含目录和文件信息。
    """
    # 安全检查：路径必须在 workspace 内
    safe_path = sandbox._safe_path(path)  # 使用私有方法（内部已有实现）
    if safe_path is None:
        return {"error": "路径越权或不存在", "tree": []}

    def _walk_dir(root: str, rel_path: str = ".") -> List[Dict]:
        """递归遍历目录，生成树结构"""
        items = []
        full_path = os.path.join(root, rel_path) if rel_path != "." else root
        try:
            for name in os.listdir(full_path):
                full_item = os.path.join(full_path, name)
                rel_item = os.path.join(rel_path, name) if rel_path != "." else name
                if os.path.isdir(full_item):
                    items.append({
                        "name": name,
                        "type": "directory",
                        "children": _walk_dir(root, rel_item)
                    })
                else:
                    size = os.path.getsize(full_item)
                    items.append({
                        "name": name,
                        "type": "file",
                        "size": size
                    })
        except Exception as e:
            items.append({"error": str(e)})
        return items

    try:
        tree = _walk_dir(safe_path)
        return {"tree": tree, "error": None}
    except Exception as e:
        return {"error": str(e), "tree": []}


def create_readonly_tool_registry(workspace: str) -> ToolRegistry:
    """创建只读工具注册表，仅包含 read_file 和 list_directory"""
    sandbox = SandboxExecutor(workspace_root=workspace)
    registry = ToolRegistry(sandbox)

    # 注册 read_file
    registry.register(
        name="read_file",
        description="读取沙箱内的文件（最大 1MB）",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件相对路径（相对于 workspace）"},
                "timeout": {"type": "number", "description": "超时秒数（默认10）", "default": 10.0},
            },
            "required": ["path"],
        },
        func=lambda path, timeout=10.0: sandbox.read_file(path, timeout)
    )

    # 注册 list_directory
    registry.register(
        name="list_directory",
        description="列出沙箱内的目录结构（递归），返回 JSON 树",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "目录相对路径（相对于 workspace），默认为 '.'", "default": "."},
            },
        },
        func=lambda path=".": list_directory(sandbox, path)
    )

    return registry


# ============================================================
# 带详细日志的 Orchestrator（展示五层执行链路）
# ============================================================

class VerboseOrchestrator(Orchestrator):
    """
    在 Orchestrator 基础上，重写 run_goal 以打印清晰的执行链路。
    """

    async def run_goal(self, goal: str) -> Dict[str, Any]:
        print("\n" + "=" * 70)
        print("📌 [Orchestrator] 接收用户目标")
        print(f"   Goal: {goal}")
        print("=" * 70)

        # 1. 调用 Planner 生成计划
        print("\n🧠 [Planner] 正在拆解目标为可执行步骤...")
        steps: List[PlanStep] = await self.planner.plan(
            goal=goal,
            available_capabilities=[]  # 不依赖具体工具描述
        )
        print(f"\n✅ [Planner] 生成 {len(steps)} 个步骤：")
        for s in steps:
            print(f"   - Step {s.step_id}: {s.description}")

        # 2. 执行计划（通过 Executor）
        print("\n⚙️ [Executor] 开始调度执行步骤...")
        results: Dict[int, Any] = {}
        step_outputs = {}  # 用于最终报告

        # 记录 event_bus 起始长度，以便提取本步骤的事件
        event_bus = self.runtime.event_bus
        prev_len = len(event_bus)

        for idx, step in enumerate(steps, 1):
            print(f"\n   ▶️  执行 Step {step.step_id}/{len(steps)}: {step.description}")

            # 记录执行前 bus 长度
            before_len = len(event_bus)

            try:
                # 调用 Executor 执行单个步骤（内部调用 runtime.run）
                result = await self.executor.execute_single_step(step)
                results[step.step_id] = result

                # 获取本步骤新增事件（用于展示工具调用）
                new_events = list(event_bus)[before_len:]
                tool_calls = [
                    e for e in new_events
                    if e.type in (EventType.TOOL_CALL_REQUEST, EventType.TOOL_CALL_RESULT)
                ]
                if tool_calls:
                    print("      🔧 [Runtime] 工具调用记录：")
                    for e in tool_calls:
                        if e.type == EventType.TOOL_CALL_REQUEST:
                            data = e.data
                            if isinstance(data, list):
                                for tc in data:
                                    print(f"         - 请求: {tc.get('name')}({json.dumps(tc.get('arguments', {}), ensure_ascii=False)})")
                        elif e.type == EventType.TOOL_CALL_RESULT:
                            data = e.data
                            if isinstance(data, dict):
                                print(f"         - 结果: {data.get('name')} -> {str(data.get('result', ''))[:100]}...")

                # 提取 final_answer 作为步骤输出
                if isinstance(result, dict) and "final_answer" in result:
                    step_outputs[step.step_id] = result["final_answer"]
                else:
                    step_outputs[step.step_id] = str(result)[:500]

                print(f"      ✅ 步骤完成，结果摘要: {str(result.get('final_answer', result))[:150]}...")

            except Exception as e:
                results[step.step_id] = {"error": str(e)}
                step_outputs[step.step_id] = f"错误: {e}"
                print(f"      ❌ 步骤失败: {e}")

        # 3. 聚合生成最终报告
        print("\n📊 [Orchestrator] 聚合所有步骤结果，生成最终报告...")
        # 将各步骤输出拼接成 Markdown 格式
        report_lines = [
            "# Mini Harness 架构审计报告\n",
            f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"**审计目标**: {goal}\n\n",
            "## 执行步骤摘要\n"
        ]
        for step_id, output in step_outputs.items():
            report_lines.append(f"### 步骤 {step_id}\n")
            report_lines.append(f"{output}\n\n")

        # 额外添加总体结论（从最后一个步骤或 summary 中提取）
        final_summary = "审计完成，请查看各步骤详情。"
        if step_outputs:
            last_output = step_outputs[max(step_outputs.keys())]
            if isinstance(last_output, str) and len(last_output) > 10:
                final_summary = last_output

        report_lines.append("## 总结\n")
        report_lines.append(final_summary)

        final_report = "\n".join(report_lines)

        print("\n✅ [Orchestrator] 报告生成完成。")

        return {
            "goal": goal,
            "plan": steps,
            "results": results,
            "summary": final_report,  # 返回完整报告
        }


# ============================================================
# Trace 导出与摘要生成
# ============================================================

def export_trace(event_bus: deque, trace_dir: str = "traces") -> str:
    """将事件总线导出为 JSON 文件，返回文件路径"""
    os.makedirs(trace_dir, exist_ok=True)
    timestamp = int(time.time() * 1000)
    trace_path = os.path.join(trace_dir, f"trace_{timestamp}.json")
    events_serializable = []
    for e in event_bus:
        events_serializable.append({
            "type": e.type.value,
            "timestamp": e.timestamp,
            "data": e.data,
            "trace_id": e.trace_id if hasattr(e, 'trace_id') else "",
        })
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(events_serializable, f, indent=2, ensure_ascii=False)
    return trace_path


def generate_trace_summary(trace_path: str) -> str:
    """从 trace JSON 生成 Execution Timeline + Statistics 摘要"""
    with open(trace_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    # 按时间排序（已经有序）
    timeline = []
    tool_call_count = 0
    llm_request_count = 0
    error_count = 0
    user_input = None

    for e in events:
        typ = e["type"]
        data = e["data"]
        if typ == "user_input":
            user_input = data
        elif typ == "llm_request":
            llm_request_count += 1
            timeline.append(f"  - LLM 请求 ({e['timestamp']})")
        elif typ == "llm_response":
            timeline.append(f"  - LLM 响应 ({e['timestamp']})")
        elif typ == "tool_call_request":
            tool_call_count += 1
            if isinstance(data, list):
                names = [tc.get("name") for tc in data]
                timeline.append(f"  - 工具调用请求: {', '.join(names)} ({e['timestamp']})")
        elif typ == "tool_call_result":
            if isinstance(data, dict):
                timeline.append(f"  - 工具结果: {data.get('name')} -> {str(data.get('result', ''))[:50]}... ({e['timestamp']})")
        elif typ == "error":
            error_count += 1
            timeline.append(f"  - ❌ 错误: {data} ({e['timestamp']})")
        elif typ == "finish":
            timeline.append(f"  - ✅ 任务完成 ({e['timestamp']})")

    # 构建摘要文本
    summary_lines = [
        "=" * 60,
        "📜 Trace Summary (Execution Timeline + Statistics)",
        "=" * 60,
        f"用户目标: {user_input or '未知'}",
        "",
        "⏱️ 执行时间线:",
    ]
    summary_lines.extend(timeline)
    summary_lines.append("")
    summary_lines.append("📊 统计信息:")
    summary_lines.append(f"  - 总 LLM 请求次数: {llm_request_count}")
    summary_lines.append(f"  - 总工具调用次数: {tool_call_count}")
    summary_lines.append(f"  - 错误次数: {error_count}")
    summary_lines.append("=" * 60)
    return "\n".join(summary_lines)


# ============================================================
# 主函数
# ============================================================

async def main():
    # 0. 设置工作目录为项目根（便于读取项目文件）
    project_root = Path(__file__).parent.parent
    workspace = str(project_root)  # 审计整个项目

    # 1. 构建只读工具注册表
    tool_registry = create_readonly_tool_registry(workspace)

    # 2. 构建 LLM 客户端（使用真实 DeepSeek API）
    llm_client = RealLLMClient()  # 从环境变量读取 DEEPSEEK_API_KEY

    # 3. 构建事件总线（共享 deque）
    event_bus = deque(maxlen=10000)

    # 4. 构建 Context Manager（注入 event_bus）
    context_manager = ContextManager(
        llm_client=llm_client,
        max_tokens=8000,
        event_bus=event_bus
    )

    # 5. 构建 Runtime 配置
    config = RuntimeConfig(
        workspace=workspace,
        max_iterations=10,
        enable_concurrent_tools=True,
        tool_timeout=30.0,
        max_context_tokens=8000,
        event_bus_maxlen=10000,
        enable_mcp=False,  # 不使用 MCP
    )

    # 6. 构建 Runtime（传入共享 event_bus 和 context_manager）
    runtime = HarnessRuntime(
        config=config,
        llm_client=llm_client,
        tool_registry=tool_registry,
        context_manager=context_manager,
        scope=None,  # 不需要 AgentScope
    )
    # 让 runtime 使用同一个 event_bus（覆盖其内部的 deque）
    runtime.event_bus = event_bus  # 确保所有事件记录到同一个 bus

    # 7. 构建 Planner（使用 LLM 辅助规划，也可纯规则）
    planner = Planner(llm_client=llm_client)

    # 8. 构建 VerboseOrchestrator
    orchestrator = VerboseOrchestrator(
        runtime=runtime,
        planner=planner,
        scope=None
    )

    # 9. 定义审计目标
    goal = """
        请作为架构审计 Agent，对 Mini Harness 项目执行一次只读的架构审计：
        1. 扫描项目结构，识别核心模块
        2. 分析 Runtime 的单一职责与依赖关系
        3. 验证 Tool Calling 链路（Runtime → MCP → Sandbox）
        4. 检查 Multi-Agent 层的隔离设计
        5. 输出架构风险报告（markdown 格式）
        约束：禁止修改任何文件，禁止执行写操作或 shell 命令。
        """.strip()

    # 10. 执行审计
    print("\n🚀 开始执行 Mini Harness 自我架构审计（只读模式）...\n")
    result = await orchestrator.run_goal(goal)

    # 11. 导出完整 Trace
    trace_path = export_trace(event_bus, "traces")
    print(f"\n📁 完整 Trace 已导出至: {trace_path}")

    # 12. 生成并打印 Trace Summary
    trace_summary = generate_trace_summary(trace_path)
    print("\n" + trace_summary)

    # 13. 保存最终报告
    report_content = result["summary"]
    report_path = "architecture_audit_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\n📄 最终审计报告已保存至: {report_path}")

    # 14. 打印最终报告（可选）
    print("\n" + "=" * 60)
    print("📄 最终审计报告预览（前 500 字符）:")
    print(report_content[:500] + ("..." if len(report_content) > 500 else ""))
    print("=" * 60)

    print("\n✅ Demo 执行完成。")


if __name__ == "__main__":
    asyncio.run(main())