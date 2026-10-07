from typing import List, Dict, Optional, Callable, Awaitable, Any
from collections import deque
from mini_harness.core.interfaces import LLMClient
from mini_harness.core.models import Event, EventType, AgentState, AgentStatus
from mini_harness.infra.tools import ToolRegistry
from mini_harness.infra.config import RuntimeConfig
from mini_harness.infra.context import ContextManager
from mini_harness.mcp.client import MCPClient
from mini_harness.agents.scope import AgentScope
import asyncio
import uuid
import random
import json
import os, json, time
import sys


# ---------- Harness Runtime 主类 ----------
class HarnessRuntime:
    """
    Mini Harness 核心调度器。
    职责：维护Event Bus、管理状态机、调度LLM和Tool执行。
    """
    
    def __init__(
            self, 
            config: RuntimeConfig,
            llm_client: LLMClient, 
            tool_registry: ToolRegistry, 
            context_manager: Optional[ContextManager] = None,   # 新增
            scope: Optional[AgentScope] = None,
    ):
        self.config = config
        self.llm = llm_client
        self.tools = tool_registry
        self.scope = scope              # Agent隔离上下文
        self.event_bus: deque = deque(maxlen=self.config.event_bus_maxlen)  # 全量事件记录
        self.state: Optional[AgentState] = None
        self._trace_id: Optional[str] = None                                # 新增

        # ===== MCP 相关属性（仅保存配置，不启动） =====
        self._mcp_client = None
        self._mcp_process = None
        self._mcp_started = False
        self._enable_mcp = getattr(self.config, 'enable_mcp', False)

        #----------------将原本的workspace改成scope---------------------
        if self.scope is not None:
            self.agent_id = self.scope.agent_id
            self.workspace = self.scope.workspace
        else:
            self.agent_id = None
            self.workspace = self.config.workspace

        # ---------- 集成 ContextManager ----------
        # 从 RuntimeConfig 获取 Context token 上限
        max_tokens = config.max_context_tokens

        if context_manager is None:
            self.context = ContextManager(
                llm_client=llm_client,
                max_tokens=max_tokens,
                event_bus=self.event_bus
            )
        else:
            self.context = context_manager
            # 若外部传入的 context_manager 尚未持有 event_bus，则注入
            if not hasattr(self.context, 'event_bus') or self.context.event_bus is None:
                self.context.event_bus = self.event_bus

        # ========== 新增：MCP 客户端初始化 ==========
        self._mcp_client: Optional[MCPClient] = None
        self._mcp_process = None

    async def close(self):
        """关闭 MCP 客户端和子进程（非必须，但建议在程序退出前调用）"""
        if self._mcp_client:
            await self._mcp_client.close()
            self._mcp_client = None

        if self._mcp_process:
            if self._mcp_process.returncode is None:
                self._mcp_process.terminate()

                try:
                    await asyncio.wait_for(
                        self._mcp_process.wait(),
                        timeout=3
                    )
                except asyncio.TimeoutError:
                    self._mcp_process.kill()
                    await self._mcp_process.wait()

            self._mcp_process = None

    def _record_event(
            self, 
            event_type: EventType, 
            data: Any = None) -> None:
        
        if self._trace_id is None:
            self._trace_id = str(uuid.uuid4())
        
        self.event_bus.append(
            Event(
                type=event_type, 
                data=data, 
                trace_id=self._trace_id
            )
        )
   
    # ========== MCP 懒加载启动 ==========
    async def _ensure_mcp_started(self):
        """异步启动 MCP Server 子进程并完成握手（仅执行一次）"""
        if self._mcp_started or not self._enable_mcp:
            return

        # 构建子进程环境变量
        env = os.environ.copy()
        # 假设项目采用 src/ 布局，根据当前文件位置向上三级得到项目根
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        src_dir = os.path.join(project_root, "src")
        if os.path.isdir(src_dir):
            pythonpath = src_dir
        else:
            # 如果不是 src/ 布局，直接使用项目根
            pythonpath = project_root
        if "PYTHONPATH" in env:
            pythonpath += os.pathsep + env["PYTHONPATH"]
        env["PYTHONPATH"] = pythonpath
        env["MCP_WORKSPACE"] = self.workspace

        # 启动子进程
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "mini_harness.mcp.server",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=project_root,
            env=env,
        )
        self._mcp_process = process
        self._mcp_client = MCPClient(process)
        await self._mcp_client.initialize()
        self._mcp_started = True

        # 后台任务读取 stderr（便于调试）
        async def read_stderr():
            while True:
                line = await process.stderr.readline()
                if not line:
                    break
                print(
                    f"[MCP Server] {line.decode().strip()}",
                    file=sys.stderr
                )
        asyncio.create_task(read_stderr())


    async def run(self, user_query: str) -> Dict[str, Any]:
        """
        核心入口：接收用户输入，执行Agent Loop，返回最终结果。
        
        输入: user_query (str) 
        输出: {
            "final_answer": str,                # 最终回复
            "iterations": int,                  # 实际循环次数
            "total_tokens": int,                # 预估Token消耗
            "events": List[Event],              # 全链路事件（用于Debug）
            "status": AgentStatus               # 最终状态
        }
        
        IO密集型强化点：
        1. 整个Loop基于 asyncio 实现异步非阻塞。
        2. 当检测到多个Tool Call时，使用 asyncio.gather() 并发执行。
        3. 每个Tool Call包裹 asyncio.wait_for(tool_task, timeout=30.0) 实现超时取消。
        """
        self._trace_id = str(uuid.uuid4())
        self.state = AgentState(max_iterations=self.config.max_iterations)
        self.context.add_user_message(user_query)
        self._record_event(EventType.USER_INPUT, user_query)

        while self.state.current_iteration < self.config.max_iterations:
            self.state.current_iteration += 1
            self._record_event(EventType.LOOP_ITERATION, self.state.current_iteration)

            should_continue = await self._step()
            if not should_continue:
                break

        if self.state.final_answer is not None:
            self.state.status = AgentStatus.FINISHED
            self._record_event(EventType.FINISH, self.state.final_answer)
        else:
            self.state.status = AgentStatus.ERROR if self.state.error_info else AgentStatus.FINISHED
            if not self.state.final_answer:
                self.state.final_answer = "Task stopped due to max iterations or error."

        # ==============================
        # 新增：等待后台Context压缩完成
        # ==============================
        if self.context._background_task:
            await self.context._background_task

        # 在返回前添加持久化
        # 导出事件到 traces/ 目录
        trace_dir = "traces"
        os.makedirs(trace_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        trace_path = os.path.join(trace_dir, f"trace_{timestamp}.json")
        with open(trace_path, "w", encoding="utf-8") as f:
            events_serializable = []
            for e in self.event_bus:
                events_serializable.append({
                    "type": e.type.value,
                    "timestamp": e.timestamp,
                    "data": e.data,
                    "trace_id": e.trace_id
                })

            json.dump(events_serializable, f, indent=2, ensure_ascii=False)

        return {
            "final_answer": self.state.final_answer,
            "iterations": self.state.current_iteration,
            "total_tokens": random.randint(100, 500),   # 仅用于演示估算
            "events": list(self.event_bus),             # 将 event_bus 转为列表以便序列化
            "status": self.state.status
        }
    
    async def _step(self) -> bool:
        """
        单步执行：调用LLM -> 判断是否要调用工具 -> 执行工具 -> 返回结果。
        集成 MCP：如果启用了 MCP，则通过 MCPClient 调用工具；否则使用原有 ToolRegistry。
        返回True表示循环继续，False表示结束或出错。
        内部通过 self.state 维护当前状态。
        """

        try:
            self.state.status = AgentStatus.THINKING
            self._record_event(EventType.AGENT_THINKING, "Calling LLM...")

            messages = self.context.get_context_for_llm()

            
            # 获取工具schema
            tools_schema = None
            if self._enable_mcp and self._mcp_client is not None:
                # MCP模式暂时先保持兼容
                # 后续可以从 MCP tools/list 动态获取 schema
                pass
            else:
                tools_schema = self.tools.get_schema()

            self._record_event(
                EventType.LLM_REQUEST,
                {
                    "messages": messages,
                    "tools": tools_schema
                }
            )
            response = await self.llm.generate(
                messages,
                tools=tools_schema
            )
            self._record_event(
                EventType.LLM_RESPONSE,
                response
            )

            content = response.get("content")
            tool_calls = response.get("tool_calls", [])
            # ============================= 第一项修补：死循环防御 =============================
            if tool_calls:
                # 1. 生成“归一化记录”（无视参数顺序，无视列表顺序）
                try:
                    # 将每个工具调用按参数key排序，再整体排序，保证 {a:1,b:2} 和 {b:2,a:1} 一致
                    sorted_calls = sorted(
                        [
                            {
                                "name": tc["name"], 
                                "args": json.dumps(tc.get("arguments", {}), sort_keys=True)
                            } 
                            for tc in tool_calls
                        ],
                        key=lambda x: x["name"] + x["args"]  # 按名字+参数排序，确保多工具时列表顺序一致
                    )
                    current_signature = json.dumps(sorted_calls, sort_keys=True)
                except Exception:
                    # 容错降级：如果解析失败，退化为纯工具名称列表
                    current_signature = str(sorted([tc.get("name") for tc in tool_calls]))

                # 2. 指纹比对（状态机快照）
                if self.state._last_tool_call_signature == current_signature:
                    self.state._consecutive_duplicate_count += 1
                    # 记录告警事件（models.py加的WARNING类型）
                    self._record_event(
                        EventType.WARNING, 
                        f"Duplicate tool call detected (#{self.state._consecutive_duplicate_count}): {current_signature}"
                    )
                else:
                    # 只要指纹变了，说明Agent换了思路，重置计数器
                    self.state._consecutive_duplicate_count = 1
                    self.state._last_tool_call_signature = current_signature
                # 3. 连续重复达到3次 -> 直接抛出RecursionError
                if self.state._consecutive_duplicate_count >= 3:
                    error_msg = f"RecursionError: Agent陷入死循环，连续3次请求相同Tool Call: {current_signature}"
                    self.state.error_info = error_msg
                    self.state.status = AgentStatus.ERROR
                    self._record_event(EventType.ERROR, error_msg)
                    raise RecursionError(error_msg)
            
            # ========== 工具执行（核心修改） ==========
            if tool_calls:
                self.state.status = AgentStatus.CALLING_TOOL
                self.state.pending_tool_calls = tool_calls

                assistant_tool_calls = []
                for tc in tool_calls:
                    assistant_tool_calls.append(
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(
                                    tc["arguments"],
                                    ensure_ascii=False
                                )
                            }
                        }
                    )
                self.context.add_assistant_tool_calls(assistant_tool_calls)

                self._record_event(EventType.TOOL_CALL_REQUEST, tool_calls)

                # ===== 确保 MCP 已启动（如果启用） =====
                if self._enable_mcp:
                    await self._ensure_mcp_started()

                tasks = []
                for tc in tool_calls:
                    name = tc["name"]
                    args = tc.get("arguments", {})

                    # 根据是否启用 MCP 选择执行方式
                    if self._enable_mcp and self._mcp_client is not None:
                        # 使用 MCPClient 调用工具
                        async def mcp_call(name, args):
                            return await self._mcp_client.call_tool(name, args)
                        task = asyncio.wait_for(
                            mcp_call(name, args),
                            timeout=self.config.tool_timeout
                        )

                    else:
                        # 原有 ToolRegistry 方式
                        task = asyncio.wait_for(
                            self.tools.execute_with_retry(name, **args), 
                            timeout=self.config.tool_timeout
                        )

                    tasks.append(task)

                self.state.status = AgentStatus.EXECUTING
                # 如果配置关闭并发，则顺序执行，但这里我们总是用gather（如果只有一个也等效）
                if self.config.enable_concurrent_tools and len(tasks) > 1:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    # 顺序执行（模拟串行）
                    results = []
                    for t in tasks:
                        try:
                            res = await t
                            results.append(res)
                        except Exception as e:
                            results.append(e)

                # 处理结果（与原有逻辑一致）
                for idx, tc in enumerate(tool_calls):
                    result = results[idx]
                    if isinstance(result, Exception):
                        # ---- 修复点：区分异常类型，记录更详细的上下文 ----
                        error_type = type(result).__name__
                        if isinstance(result, asyncio.TimeoutError):
                            error_msg = f"Timeout after {self.config.tool_timeout}s"
                        else:
                            error_msg = str(result)
                        result_str = f"{error_type}: {error_msg}"
                        self._record_event(
                            EventType.ERROR, 
                            {
                                "tool_name": tc["name"],
                                "args": tc.get("arguments", {}),  # 记录参数！方便复现
                                "error_type": error_type,
                                "error_msg": error_msg,
                                "iteration": self.state.current_iteration   
                            }
                        )
                    else:
                        result_str = str(result)
                        
                    self._record_event(
                        EventType.TOOL_CALL_RESULT, 
                        {
                            "name": tc["name"], 
                            "result": result_str
                        }
                    )
                    # 然后添加到 ContextManager（注意不需要 tool_call_id，ContextManager 只存储 role 和 content）
                    self.context.add_tool_result(
                        tool_call_id=tc["id"],
                        result=result_str
                    )
                    # TODO:
                    #         ContextManager.add_tool_result()
                    #         需要支持 tool_call_id
                    #         否则真实Function Calling兼容性不足
                self.state.pending_tool_calls = []
                return True

            else:
                if content is not None:
                    self.state.final_answer = content
                    self.context.add_assistant_message(content)
                else:
                    self.state.error_info = "LLM returned empty response without tool calls."
                    self._record_event(EventType.ERROR, self.state.error_info)
                return False

        except Exception as e:
            #调试打印：
            print(
                "========== RUNTIME STEP ERROR ==========",
                flush=True
            )

            print(
                type(e).__name__,
                str(e),
                flush=True
            )

            print(
                "========================================",
                flush=True
            )
            self.state.error_info = f"Step failed: {str(e)}"
            self.state.status = AgentStatus.ERROR
            self._record_event(EventType.ERROR, self.state.error_info)
            return False
    
    def get_trace(self) -> List[Event]:
        """获取全量事件轨迹（给Day 9的可观测性打底）"""
        return self.event_bus