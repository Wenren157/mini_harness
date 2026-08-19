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
import fnmatch
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

# ===== 新增：敏感文件黑名单 =====
SENSITIVE_PATTERNS = [
    ".env", "*.key", "*.pem", "*.p12", "*.crt",
    "id_rsa", "id_ecdsa", ".git/config", ".git/HEAD"
]

# ============================================================
# 只读工具定义
# ============================================================

async def list_directory(sandbox: SandboxExecutor, path: str = ".") -> Dict[str, Any]:
    """
    列出沙箱内指定目录的树形结构（只读）。
    返回 JSON 字符串，包含目录和文件信息。
    """
    safe_path = sandbox._safe_path(path)
    if safe_path is None:
        return {"error": "路径越权或不存在", "tree": []}

    def _walk_dir(root: str, rel_path: str = ".") -> List[Dict]:
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


def create_readonly_tool_registry(
        workspace: str,
        allowed_prefixes: Optional[List[str]] = None,
        success_list: Optional[List[str]] = None   # 新增参数
) -> ToolRegistry:
    """创建只读工具注册表，仅包含 read_file 和 list_directory"""
    sandbox = SandboxExecutor(workspace_root=workspace)
    registry = ToolRegistry(sandbox)

    def _is_allowed_path(path: str) -> bool:
        if not allowed_prefixes:
            return True
        abs_path = os.path.abspath(os.path.join(workspace, path))
        if not abs_path.startswith(os.path.abspath(workspace)):
            return False
        rel_path = os.path.relpath(abs_path, workspace).replace('\\', '/')
        for prefix in allowed_prefixes:
            prefix = prefix.replace('\\', '/')
            if rel_path == prefix or rel_path.startswith(prefix + '/'):
                return True
        return False

    async def safe_read_file(path: str, timeout: float = 10.0):
        # 检查敏感模式
        for pattern in SENSITIVE_PATTERNS:
            if fnmatch.fnmatch(path, pattern) or \
               fnmatch.fnmatch(os.path.basename(path), pattern):
                return {"content": "", "size": 0, "error": f"敏感文件，禁止读取: {path}"}
        # 白名单检查
        if not _is_allowed_path(path):
            return {"content": "", "size": 0, "error": f"路径不在允许范围内: {path}"}
        result = await sandbox.read_file(path, timeout)
        # 如果成功，记录路径
        if isinstance(result, dict) and not result.get("error"):
            if success_list is not None:
                success_list.append(path)
        return result

    # 注册 read_file
    registry.register(
        name="read_file",
        description="读取沙箱内的文件（最大 1MB）",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件相对路径（相对于 workspace）"
                },
                "timeout": {
                    "type": "number",
                    "description": "超时秒数（默认10）",
                    "default": 10.0
                },
            },
            "required": ["path"],
        },
        func=safe_read_file
    )

    # 注册 list_directory
    registry.register(
        name="list_directory",
        description="列出沙箱内的目录结构（递归），返回 JSON 树",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "目录相对路径（相对于 workspace），默认为 '.'",
                    "default": "."
                },
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
    def __init__(
            self,
            runtime: HarnessRuntime,
            planner: Planner,
            scope: Optional[AgentScope] = None,
            success_reads: Optional[List[str]] = None   # 新增参数
    ):
        super().__init__(runtime, planner, scope)
        self.original_goal = None
        self.success_reads = success_reads if success_reads is not None else []

    async def run_goal(self, goal: str) -> Dict[str, Any]:
        self.original_goal = goal
        print("\n" + "=" * 70)
        print("📌 [Orchestrator] 接收用户目标")
        print(f"   Goal: {goal}")
        print("=" * 70)

        # 1. 调用 Planner 生成计划
        print("\n🧠 [Planner] 正在拆解目标为可执行步骤...")
        context_msg = """
            【角色】你是一位拥有10年经验的分布式系统架构师，专精于 Agent Runtime 和 AI Infra。
            【项目背景】Mini Harness 是一个用于面试展示的最小 Agent Runtime，从 Day1 到 Day7 逐步实现：
            - Day1-4：Runtime、Tool Calling、Context、EventBus、Trace、Memory。
            - Day5：引入 MCP（stdio JSON-RPC），完成真机 LLM 验证。
            - Day6：实现 Multi-Agent（Planner + Executor + Orchestrator + AgentScope + MessageBus）。
            - 项目定位是“最小实现”，因此一些复杂特性（向量检索、DAG 调度、分布式存储）被刻意省略，用 # TODO 标记。
            请基于以上背景，评估当前实现的合理性，指出哪些省略是合理的，哪些遗漏是风险。

            【审计任务拆解】请将审计任务拆解为以下子任务，每个子任务必须包含明确的检查点：
            1. 核心 Runtime 审计：
            - Agent Loop 的状态机是否完备？有无遗漏状态？
            - 并发工具执行（asyncio.gather）是否安全？有无竞态？
            - 超时控制是否真正有效？超时后资源是否被正确清理？
            - 死循环防御的指纹比对算法是否可靠？
            2. 工具与沙箱审计：
            - 路径越权防护（_safe_path）是否覆盖所有文件操作？
            - 文件读写大小限制是否可绕过？
            - Shell 执行是否可能注入恶意命令？
            - 子进程超时后是否被强制杀死？
            3. 上下文与记忆审计：
            - 上下文压缩触发条件是否合理？压缩是否可能丢失关键信息？
            - 异步压缩与主循环的并发安全如何保证？
            - LRU 淘汰策略是否符合实际使用场景？
            4. MCP 协议审计：
            - JSON-RPC 通信是否处理了超时、断连、部分响应？
            - 工具 schema 是否与 Runtime 契约一致？
            - 子进程生命周期管理是否完备？
            5. Multi-Agent 审计：
            - Orchestrator 是否真正隔离了 Runtime？
            - AgentScope 的隔离粒度是否足够？
            - 如果两个 Agent 并发修改同一文件，如何避免冲突？
            6. 生产化差距分析：
            - 与真实生产级 Agent Infra 相比，缺少哪些关键组件？
            - 性能瓶颈可能出现在哪里？如何优化（如内存池化、零拷贝、连接复用）？
            - 可观测性还缺什么（如指标监控、告警、分布式追踪）？

            【工具使用限制】优先读取以下核心文件，并关注关键函数：
            - core/runtime.py：Agent Loop、状态机、工具调用、死循环防御
            - infra/tools.py：SandboxExecutor 的路径防护、超时、文件大小限制
            - infra/context.py：滑动窗口、异步压缩、并发锁
            - mcp/client.py & server.py：子进程管理、JSON-RPC 通信
            - agents/orchestrator.py：聚合逻辑、报告生成
            对于每个核心文件，至少提出一个可改进点。
        """
        steps: List[PlanStep] = await self.planner.plan(
            goal=goal,
            available_capabilities=[],
            context=context_msg
        )
        print(f"\n✅ [Planner] 生成 {len(steps)} 个步骤：")
        for s in steps:
            print(f"   - Step {s.step_id}: {s.description}")

        # 2. 执行计划（通过 Executor）
        print("\n⚙️ [Executor] 开始调度执行步骤...")
        results: Dict[int, Any] = {}
        event_bus = self.runtime.event_bus

        for idx, step in enumerate(steps, 1):
            print(f"\n   ▶️  执行 Step {step.step_id}/{len(steps)}: {step.description}")
            before_len = len(event_bus)

            try:
                result = await self.executor.execute_single_step(step)
                results[step.step_id] = result

                # 展示工具调用
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

                print(f"      ✅ 步骤完成，结果摘要: {str(result.get('final_answer', result))[:150]}...")

            except Exception as e:
                results[step.step_id] = {"error": str(e)}
                print(f"      ❌ 步骤失败: {e}")

        # 3. 聚合压缩：调用 LLM 生成精简报告
        print("\n📊 [Orchestrator] 压缩聚合结果，生成精简审计报告...")
        step_summaries = []
        for step_id, result in results.items():
            if isinstance(result, dict) and "final_answer" in result:
                step_summaries.append(f"步骤{step_id}: {result['final_answer'][:200]}...")
            else:
                step_summaries.append(f"步骤{step_id}: {str(result)[:200]}...")

        # 动态构建已读取文件列表
        if self.success_reads:
            read_files_str = "\n".join(f"  - `{f}`" for f in self.success_reads)
        else:
            read_files_str = "（无成功读取的文件）"

        summary_prompt = f"""
            【角色】你是一位拥有10年经验的分布式系统架构师，正在对 Mini Harness 项目进行严格的架构审计。

            **重要事实：你已成功读取以下文件（基于实际工具调用结果）：**
            {read_files_str}

            因此，你的分析**必须严格基于**这些实际读取的文件内容。如果某风险无法从已读取文件中找到证据，则不要写入报告。

            请基于以下各步骤的审计结果，生成最终的架构审计报告。
            报告必须包含：
            1. 执行摘要（150字以内）
            2. 架构优点（最多5条，简明）
            3. 关键风险清单（至少5条，按严重度分级：高危/中危/低危）
            - 每条必须包含：风险描述、证据（文件路径+行号）、影响、建议修复方案
            4. 性能优化建议（至少3条，指出具体瓶颈和优化方向）
            5. 生产就绪度差距分析（列出至少4项缺失的生产级能力，并说明为何重要）
            6. 总结论

            总字数不超过1500字。禁止重复各步骤的原始内容，只输出提炼后的关键发现。
            每个风险证据路径必须来自上述成功读取的文件列表，禁止编造。
            如果某个风险没有实际读取到对应文件，则不要写入报告。

            原始目标：{goal}
            各步骤结果摘要：
            {chr(10).join(step_summaries)}
            """
        try:
            llm_response = await self.runtime.llm.generate([{"role": "user", "content": summary_prompt}])
            final_report = llm_response.get("content", "报告生成失败")
        except Exception as e:
            final_report = f"报告生成失败（降级拼接）：\n" + "\n".join(step_summaries)

        print("\n✅ [Orchestrator] 精简报告生成完成。")

        return {
            "goal": goal,
            "plan": steps,
            "results": results,
            "summary": final_report,
        }


# ============================================================
# Trace 导出与摘要生成
# ============================================================

def export_trace(event_bus: deque, trace_dir: str = "traces") -> str:
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


def generate_trace_summary(trace_path: str, goal: Optional[str] = None) -> str:
    with open(trace_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    user_input = goal
    if not user_input:
        for e in events:
            if e.get("type") == "user_input":
                user_input = e.get("data", "未知")
                break

    if not user_input:
        user_input = "未知"

    timeline = []
    tool_call_count = 0
    llm_request_count = 0
    error_count = 0

    for e in events:
        typ = e["type"]
        data = e["data"]
        if typ == "llm_request":
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

    summary_lines = [
        "=" * 60,
        "📜 Trace Summary (Execution Timeline + Statistics)",
        "=" * 60,
        f"用户目标: {user_input}",
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
    project_root = Path(__file__).parent.parent
    workspace = str(project_root)

    # 1. 构建只读工具注册表
    allowed_scan_paths = ["src/mini_harness", "demo"]
    successful_reads = []   # 收集成功读取的文件
    tool_registry = create_readonly_tool_registry(
        workspace,
        allowed_prefixes=allowed_scan_paths,
        success_list=successful_reads   # 传入
    )

    llm_client = RealLLMClient()
    event_bus = deque(maxlen=10000)
    context_manager = ContextManager(
        llm_client=llm_client,
        max_tokens=8000,
        event_bus=event_bus
    )

    config = RuntimeConfig(
        workspace=workspace,
        max_iterations=10,
        enable_concurrent_tools=True,
        tool_timeout=30.0,
        max_context_tokens=8000,
        event_bus_maxlen=10000,
        enable_mcp=False,
    )

    runtime = HarnessRuntime(
        config=config,
        llm_client=llm_client,
        tool_registry=tool_registry,
        context_manager=context_manager,
        scope=None,
    )
    runtime.event_bus = event_bus

    planner = Planner(llm_client=llm_client)

    orchestrator = VerboseOrchestrator(
        runtime=runtime,
        planner=planner,
        scope=None,
        success_reads=successful_reads   # 传入
    )

    goal = """
        请作为架构审计 Agent，对 Mini Harness 项目执行一次只读的架构审计：
        1. 扫描项目结构，识别核心模块
        2. 分析 Runtime 的单一职责与依赖关系
        3. 验证 Tool Calling 链路（Runtime → MCP → Sandbox）
        4. 检查 Multi-Agent 层的隔离设计
        5. 输出架构风险报告（markdown 格式）
        约束：禁止修改任何文件，禁止执行写操作或 shell 命令。
        """.strip()

    print("\n🚀 开始执行 Mini Harness 自我架构审计（只读模式）...\n")
    result = await orchestrator.run_goal(goal)

    trace_path = export_trace(event_bus, "traces")
    print(f"\n📁 完整 Trace 已导出至: {trace_path}")

    trace_summary = generate_trace_summary(trace_path, goal=orchestrator.original_goal)
    print("\n" + trace_summary)

    report_content = result["summary"]
    report_path = "architecture_audit_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\n📄 最终审计报告已保存至: {report_path}")

    print("\n" + "=" * 60)
    print("📄 最终审计报告预览（前 500 字符）:")
    print(report_content[:500] + ("..." if len(report_content) > 500 else ""))
    print("=" * 60)

    print("\n✅ Demo 执行完成。")


if __name__ == "__main__":
    asyncio.run(main())