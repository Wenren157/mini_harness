from typing import List, Dict, Optional, Callable, Awaitable, Any
from mini_harness.core.models import Event, EventType, AgentState, AgentStatus
from mini_harness.infra.tools import ToolRegistry
from mini_harness.infra.config import RuntimeConfig
import asyncio
import uuid
import random
import json
import tiktoken

_tokenizer_cache = None

def _get_tokenizer():
    """懒加载并缓存 tokenizer，避免每次调用都重新初始化（性能优化）"""
    global _tokenizer_cache
    if _tokenizer_cache is None: # 只在第一次调用时初始化
        try:
            # DeepSeek / GPT-4 使用 cl100k_base 编码，与 Open AI 兼容
            _tokenizer_cache = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            # 兜底：如果加载失败，返回 None，后续降级为 len//3
            print(f"Warning: Failed to load tiktoken: {e}")
            _tokenizer_cache = False  # 标记为失败，避免重复尝试

    return _tokenizer_cache if _tokenizer_cache is not False else None

# ==========================第三项修补：token精分器=================================
def _estimate_tokens(text: str) -> int:
    """
    估算文本的 Token 数量。
    优先使用 tiktoken 精确计数，降级方案为 len(text) // 3。
    """
    tokenizer = _get_tokenizer()
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            # 编码失败时降级
            pass
    # 降级方案：保守估算（中英文混合约 1 token ≈ 2.5 字符，这里取 3 更保守）
    return len(text) // 3

# ---------- LLM 客户端接口 ----------
class LLMClient:
    """封装大模型调用（支持流式/非流式）"""
    
    async def generate(
            self, 
            messages: List[Dict[str, str]], 
            tools: Optional[List[Dict]] = None) -> Dict:
        """
        调用LLM，返回响应。
        返回格式必须包含：
        {
            "content": "文本回复" 或 None（如果有工具调用）,
            "tool_calls": [{"name": "read_file", "arguments": {"path": "/tmp/a.txt"}}, ...] 或 []
        }
        """
        pass

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
            ):
        self.config = config
        self.llm = llm_client
        self.tools = tool_registry
        self.event_bus: List[Event] = []        # 全量事件记录
        self.state: Optional[AgentState] = None

    def _record_event(
            self, 
            event_type: EventType, 
            data: Any = None) -> None:
        
        self.event_bus.append(
            Event(
                type=event_type, 
                data=data, 
                trace_id=self._trace_id
            )
        )
    # =======================LRU策略用于淘汰最老的轮次=====================================
    def _truncate_context(self) -> None:
        """
        滑动窗口裁剪：当上下文超过阈值时，优先删除最老的中间轮次。
        策略：永远保留 System Prompt（如果存在） 和 最近2轮对话。
        """
        max_tokens = self.config.max_context_tokens   # 从配置中读取
        # 1. 先将 messages 序列化为字符串（这一步逃不掉，因为后面精算也要用）
        total_text = json.dumps(self.state.messages, ensure_ascii=False)


        # ======================== 新增：懒加载性能提升  ===========================
            # （核心）BPE 分词的基本原理：绝大多数正常文本（英文、中文、代码）的字符数 >= Token 数。
            # 英文：1 Token ≈ 4 字符；中文：1 Token ≈ 1.5~2 字符。
            # 因此，如果 len(text) <= max_tokens，那么 Token 数 100% <= max_tokens。
            # 此时直接返回，完全无需加载 tiktoken
        if len(total_text) <= max_tokens:
            return  
        
        # 2. 只有通过粗筛（文本确实较长），才加载 tiktoken 做精确计算
        estimated_tokens = _estimate_tokens(total_text)
        
        if estimated_tokens <= max_tokens:
            return  # 安全范围内，不动
        
        # 2. 触发裁剪（类似于OS的页面置换）
        self._record_event(
            EventType.WARNING, 
            f"Context overload ({estimated_tokens} tokens), triggering sliding window truncation."
        )
        
        # ===== 构建保护集合 =====
        protected_indices = set()

        # 1.保护系统指令（若第一条是system）
        if self.state.messages and self.state.messages[0].get("role") == "system":
            protected_indices.add(0)  # 保留系统提示词
            
        # 2.保护用户原始输入：第一条 role == "user" 的消息）
        for idx, msg in enumerate(self.state.messages):
            if msg.get("role") == "user":
                protected_indices.add(idx)
                break   # 只保护第一条 user 消息
        
        # 3.保留最后2条消息（确保当前轮次不丢）
        last_two_start = max(0, len(self.state.messages) - 2)
        for i in range(last_two_start, len(self.state.messages)):
            protected_indices.add(i)
        # ===== 安全删除循环（带熔断保护） =====
        max_attempts = 100  # 防止无限循环
        attempts = 0
        
        # 只要超标，就从保护范围外删除最旧的一条
        while estimated_tokens > max_tokens and attempts < max_attempts:
            attempts += 1
            deleted_any = False

            # 找第一个可以删除的索引（不在保护集合内）
            for idx, msg in enumerate(self.state.messages):
                if idx not in protected_indices:
                    # 如果只剩保护消息，强制退出
                    # if len(self.state.messages) <= len(protected_indices):
                    #     break
                    # 删除该消息（注意pop后索引会变，所以break后重新循环）
                    removed = self.state.messages.pop(idx)
                    self._record_event(EventType.WARNING, 
                                       f"Dropped old context: {str(removed)[:50]}..."
                    )
                    deleted_any = True
                    # 更新估算
                    total_text = json.dumps(self.state.messages, ensure_ascii=False)
                    estimated_tokens = _estimate_tokens(total_text)
                    break  # 跳出for，重新while检测

            # 如果遍历完所有消息都没有可删除的，强制退出
            if not deleted_any:
                self._record_event(EventType.WARNING, 
                                   "No deletable messages found, forcing exit."
                )
                break
        
         # 如果超过尝试次数，强制退出
        if attempts >= max_attempts:
            self._record_event(EventType.ERROR, 
                               "Truncation exceeded max attempts, giving up."
            )
    
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
        self.state.messages.append({"role": "user", "content": user_query})
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

        return {
            "final_answer": self.state.final_answer,
            "iterations": self.state.current_iteration,
            "total_tokens": random.randint(100, 500),  # 仅用于演示估算
            "events": self.event_bus,
            "status": self.state.status
        }
    
    async def _step(self) -> bool:
        """
        单步执行：调用LLM -> 判断是否要调用工具 -> 执行工具 -> 返回结果。
        返回True表示循环继续，False表示结束或出错。
        内部通过 self.state 维护当前状态。
        """
        # ===== 第二项修补：执行上下文裁剪（防止显存泄漏） =====
        self._truncate_context()  # 已将max_tokens=4000设为config参数

        try:
            self.state.status = AgentStatus.THINKING
            self._record_event(EventType.AGENT_THINKING, "Calling LLM...")

            self._record_event(EventType.LLM_REQUEST, {"messages": self.state.messages})
            response = await self.llm.generate(self.state.messages)
            self._record_event(EventType.LLM_RESPONSE, response)

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
            
            # ========== 原有的执行工具代码（if tool_calls: ...）完全不动 ==========
            if tool_calls:
                self.state.status = AgentStatus.CALLING_TOOL
                self.state.pending_tool_calls = tool_calls
                self._record_event(EventType.TOOL_CALL_REQUEST, tool_calls)

                tasks = []
                for tc in tool_calls:
                    name = tc["name"]
                    args = tc.get("arguments", {})
                    # 使用配置的超时时间
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
                            {"name": tc["name"], "result": result_str}
                        )

                    self.state.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{idx}"),
                        "content": result_str
                    })
                    self._record_event(EventType.TOOL_CALL_RESULT, {"name": tc["name"], "result": result_str})

                self.state.pending_tool_calls = []
                return True

            else:
                if content is not None:
                    self.state.final_answer = content
                    self.state.messages.append({"role": "assistant", "content": content})
                else:
                    self.state.error_info = "LLM returned empty response without tool calls."
                    self._record_event(EventType.ERROR, self.state.error_info)
                return False

        except Exception as e:
            self.state.error_info = f"Step failed: {str(e)}"
            self.state.status = AgentStatus.ERROR
            self._record_event(EventType.ERROR, self.state.error_info)
            return False
    
    def get_trace(self) -> List[Event]:
        """获取全量事件轨迹（给Day 9的可观测性打底）"""
        return self.event_bus