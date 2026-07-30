from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import asyncio
import time
import re
import tiktoken

# ============================================================
# 1. Token 估算器
# ============================================================
class TokenEstimator:
    """
    Token估算器
    优先使用tiktoken精确计算
    """
    _tokenizer_cache = None

    @classmethod
    def _get_tokenizer(cls):
        """懒加载并缓存 tokenizer，避免每次调用都重新初始化（性能优化）"""
        if cls._tokenizer_cache is None:
            try:
                cls._tokenizer_cache = (
                    tiktoken.get_encoding(
                        "cl100k_base"
                    )
                )
            except Exception:
                cls._tokenizer_cache = False
        
        return (
            cls._tokenizer_cache
            if cls._tokenizer_cache is not False
            else None
        )

    @classmethod
    def estimate(cls, text: str) -> int:

        tokenizer = cls._get_tokenizer()

        if tokenizer:
            try:
                return len(
                    tokenizer.encode(text)
                )
            except Exception:
                pass
        # fallback
        return len(text)//3
    
# ============================================================
# 2. 上下文窗口
# ============================================================
@dataclass
class ContextWindow:
    """管理当前会话的上下文窗口，支持滑动裁剪"""

    max_tokens: int = 8000              # 模型上下文上限
    reserve_tokens: int = 1000          # 为 system/instruction 预留
    messages: List[Dict[str, str]] = field(default_factory=list)
    total_tokens: int = 0

    def add_message(self, role: str, content: str) -> None:
        """添加一条消息，更新 token 计数"""
        tokens = TokenEstimator.estimate(content)
        self.messages.append({"role": role, "content": content})
        self.total_tokens += tokens
    
    def get_current_tokens(self) -> int:
        return self.total_tokens
    
    def is_overflow(self) -> bool:
        """判断是否超过上下文限制（含预留）"""
        return self.total_tokens > (self.max_tokens - self.reserve_tokens)
    
    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """
        返回当前消息列表。
        如果超限，自动裁剪最早的非 system 消息，直到低于阈值。
        裁剪会修改内部状态（messages 和 total_tokens）。
        """
        if not self.is_overflow():
            return self.messages.copy()

        # 保留 system 消息
        system_msgs = [m for m in self.messages if m["role"] == "system"]
        non_system = [m for m in self.messages if m["role"] != "system"]

        # 从前往后删除非 system 消息，直到总 token 数低于阈值
        target_tokens = self.max_tokens - self.reserve_tokens
        kept_non_system = []
        kept_tokens = sum(TokenEstimator.estimate(m["content"]) for m in system_msgs)

         # 从后往前累积保留（保留最新消息）
        for msg in reversed(non_system):
            msg_tokens = TokenEstimator.estimate(msg["content"])
            if kept_tokens + msg_tokens > target_tokens:
                # 如果加上这条消息会超限，则停止（不保留这条及之前的）
                break
            kept_non_system.append(msg)
            kept_tokens += msg_tokens
        
        # 恢复顺序（从旧到新）
        kept_non_system.reverse()

        # 更新内部状态
        self.messages = system_msgs + kept_non_system
        self.total_tokens = kept_tokens
        return self.messages.copy()
    
# ============================================================
# 3. 上下文管理器（核心调度器）
# ============================================================
class ContextManager:
    """
    负责管理整个对话上下文：
    1. Token 预算监控
    2. 滑动窗口裁剪（同步，立即生效）
    3. 异步后台压缩（调用 LLM 总结历史，不阻塞主循环）
    """
    def __init__(
            self, 
            llm_client, 
            max_tokens: int = 8000,
            event_bus=None
        ):

        self.llm = llm_client
        self.event_bus = event_bus   # 新增：保存 EventBus 引用
        self.window = ContextWindow(max_tokens=max_tokens)
        self.summary: Optional[str] = None
        self._compression_lock = asyncio.Lock()
        self._background_task: Optional[asyncio.Task] = None
        self._compression_triggered = False

    def add_user_message(self, content: str) -> None:
        """添加用户消息，自动触发 Token 检查"""
        self.window.add_message("user", content)
        # TODO: 接入 event_bus 记录此事件，应记录操作类型、消息内容长度、当前 token 总数
        if self.window.is_overflow():
            self._trigger_async_compression()

    def add_assistant_message(self, content: str) -> None:
        self.window.add_message("assistant", content)
        # TODO: 接入 event_bus 记录此事件
        if self.window.is_overflow():
            self._trigger_async_compression()

    def add_tool_result(self, tool_name: str, result: str) -> None:
        self.window.add_message("tool", f"[{tool_name}] {result}")
        # TODO: 接入 event_bus 记录此事件
        if self.window.is_overflow():
            self._trigger_async_compression()

    def add_system_message(self, content: str) -> None:
        """添加 system 消息（不会被裁剪）"""
        self.window.messages.insert(0, {"role": "system", "content": content})
        self.window.total_tokens += TokenEstimator.estimate(content)
        # TODO: 接入 event_bus 记录此事件

    def _trigger_async_compression(self) -> None:
        """触发异步后台压缩（不阻塞）"""
        if self._background_task is None or self._background_task.done():
            self._background_task = asyncio.create_task(self._compress_background())
            # TODO: 接入 event_bus 记录压缩触发事件（包含当前 token 数、消息数量）

    async def _compress_background(self) -> None:
        """
        强化点：
        1. 后台异步执行，不阻塞主循环。
        2. 使用 asyncio.Lock 防止并发压缩。
        3. 压缩完成后插入 summary，裁剪旧消息。
        """
        async with self._compression_lock:
            # 防止短时间内多次触发
            if self._compression_triggered:
                return
            self._compression_triggered = True

            try:
                # TODO: 接入 event_bus 记录压缩开始事件（含当前消息数、token 数）

                # 1. 获取需要压缩的消息（排除 system）
                messages_to_summarize = [
                    m for m in self.window.messages
                    if m["role"] != "system"
                ]
                if len(messages_to_summarize) <= 3:
                    return
                
                # 2. 取最近 5 条非 system 消息进行总结
                recent_5 = messages_to_summarize[-5:]
                summary_prompt = "请用一段话（不超过50字）总结以下对话的核心内容："
                # 构造 LLM 输入
                summary_messages = [
                    {"role": "system", "content": summary_prompt},
                    {"role": "user", "content": str(recent_5)}   # 简单拼接
                ]

                try:
                    # TODO: 接入 event_bus 记录 LLM 请求（REQUEST 事件）
                    response = await self.llm.generate(summary_messages)
                    # TODO: 接入 event_bus 记录 LLM 响应成功（SUCCESS 事件）
                    summary = response.get("content", "对话摘要生成失败")
                except Exception as e:
                    # TODO: 接入 event_bus 记录异常（ERROR 事件）
                    summary = f"[压缩失败: {e}]"

                # 3. 更新 summary（保留最近一次）
                self.summary = summary

                # 4. 裁剪消息：只保留 system + 最近 3 条非 system
                system_msgs = [m for m in self.window.messages if m["role"] == "system"]
                non_system = [m for m in self.window.messages if m["role"] != "system"]
                kept = non_system[-3:]  # 保留最近 3 条

                # 5. 重建窗口：system + summary + 最近消息
                new_messages = system_msgs.copy()
                if self.summary:
                    new_messages.append({"role": "system", "content": f"[历史摘要] {self.summary}"})
                new_messages.extend(kept)

                # 6. 重新计算 token
                self.window.messages = new_messages
                self.window.total_tokens = sum(
                    TokenEstimator.estimate(m["content"]) for m in new_messages
                )

                # TODO: 接入 event_bus 记录压缩完成事件（含新 token 数、压缩后消息数）
            finally:
                self._compression_triggered = False

    def get_context_for_llm(self) -> List[Dict[str, str]]:
        """获取最终发给 LLM 的消息列表（包含 summary + 最近消息）"""
        # 同步裁剪（如果超限）
        return self.window.get_messages_for_llm()
    
# ============================================================
# 4. Prompt Builder
# ============================================================
class PromptBuilder:
    """构建系统提示词 + 动态指令"""

    def __init__(self, system_prompt: str = "", workspace_root: str = "."):
        self.system_prompt = system_prompt
        self.workspace_root = workspace_root
        self.instructions: List[str] = []
        self._context_cache: Optional[str] = None

    def add_instruction(self, instruction: str) -> None:
        self.instructions.append(instruction)
        self._context_cache = None  # 清除缓存

    def build(self) -> str:
        """拼接完整的 system prompt + instructions + workspace 上下文"""
        if self._context_cache is not None:
            return self._context_cache
        
        parts = []
        if self.system_prompt:
            parts.append(f"# System\n{self.system_prompt}")
        if self.instructions:
            parts.append(f"# Instructions\n" + "\n".join(self.instructions))
        parts.append(f"# Workspace\n{self.workspace_root}")

        self._context_cache = "\n\n".join(parts)
        return self._context_cache


