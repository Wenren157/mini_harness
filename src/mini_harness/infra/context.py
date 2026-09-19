from dataclasses import dataclass, field
from mini_harness.core.models import Event, EventType
from typing import List, Dict, Optional, Any
import asyncio
import json
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

        if text is None:
            return 0

        tokenizer = cls._get_tokenizer()
        if tokenizer:
            try:
                return len(tokenizer.encode(text))
            except Exception:
                pass
        # fallback
        return len(text)//3

    @classmethod
    def estimate_message(
        cls,
        message: Dict[str, Any],
    ) -> int:
        """
        按完整 message 估算 token。

        Tool Calling message 即使 content=None，
        tool_calls / tool_call_id / arguments 等字段
        仍然属于真实模型上下文。
        """

        serialized = json.dumps(
            message,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

        return cls.estimate(serialized)



# ============================================================
# 2. Message Block Helpers
# ============================================================
def _build_message_blocks(
    messages: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """
    将 messages 按协议关系组织成不可拆分的 Message Block。
    普通消息：
        [user]
        [assistant]

    Tool Calling：
        [
            assistant(tool_calls),
            tool,
            tool,
            ...
        ]

    目的：
    所有 Context boundary operation 都只能在 block 之间切分，
    不能把 assistant(tool_calls) 与对应 tool result(s) 拆开。

    当前使用者：
    1. ContextManager compression；
    2. ContextWindow hard trim（P1 后续步骤接入）。

    注意：
    本 helper 只维护已有合法消息的原子性，
    不负责修复已经损坏的 Context，
    因此它不是 Protocol Guard。
    """
    blocks: List[List[Dict[str, Any]]] = []
    index = 0

    while index < len(messages):
        message = messages[index]

        # --------------------------------------------------
        # Tool Calling block:
        #
        # assistant(tool_calls)
        # tool
        # tool
        # ...
        # --------------------------------------------------
        if (
            message.get("role") == "assistant"
            and message.get("tool_calls")
        ):
            block = [message]

            tool_call_ids = {
                tool_call.get("id")
                for tool_call in message.get("tool_calls", [])
                if tool_call.get("id") is not None
            }

            next_index = index + 1

            while next_index < len(messages):
                next_message = messages[next_index]

                if next_message.get("role") != "tool":
                    break

                if (
                    next_message.get("tool_call_id")
                    not in tool_call_ids
                ):
                    break

                block.append(next_message)
                next_index += 1

            blocks.append(block)
            index = next_index
            continue

        # --------------------------------------------------
        # 普通 message 自己构成一个 block。
        #
        # 如果这里出现 orphan tool，
        # 不在本层偷偷修复或删除。
        # Message Block 负责 boundary atomicity，
        # 不负责 Protocol Guard。
        # --------------------------------------------------
        blocks.append([message])
        index += 1

    return blocks

def _flatten_message_blocks(
    blocks: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    将 Message Blocks 恢复成标准 messages 列表。
    """
    return [
        message
        for block in blocks
        for message in block
    ]

# ============================================================
# 3. 上下文窗口
# ============================================================
@dataclass
class ContextWindow:
    """管理当前会话的上下文窗口，支持滑动裁剪"""

    max_tokens: int = 8000              # 模型上下文上限
    reserve_tokens: int = 1000          # 为 system/instruction 预留
    compression_headroom: int = 1000

    messages: List[Dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0

    @property
    def hard_limit(self) -> int:
        """真正允许 Context 使用的最大 token 预算。"""
        return self.max_tokens - self.reserve_tokens

    @property
    def soft_limit(self) -> int:
        """
        异步 compression 的提前触发阈值。
        在真正进入 hard overflow 前，预留 compression_headroom 作为压缩缓冲区。
        """
        return max(0, self.hard_limit - self.compression_headroom)

    def add_message_dict(
        self,
        message: Dict[str, Any],
        *,
        prepend: bool = False,
    ) -> None:
        """
        ContextWindow 的统一消息入口。
        """
        tokens = (TokenEstimator.estimate_message(message))
        if prepend:
            self.messages.insert(0, message)
        else:
            self.messages.append(message)
        self.total_tokens += tokens

    def add_message(self, role: str, content: str) -> None:
        """普通文本消息兼容入口"""
        self.add_message_dict({"role": role, "content": content})

    def recalculate_total_tokens(self) -> int:
        """
        根据当前完整 messages 重新计算 token 总量。
        用于 compression / trimming 等批量重建窗口后，
        防止 total_tokens 与真实 messages 状态失去同步。
        """
        self.total_tokens = sum(
            TokenEstimator.estimate_message(message) 
            for message in self.messages
        )
        return self.total_tokens
    
    def get_current_tokens(self) -> int:
        return self.total_tokens

    def should_compress(self) -> bool:
        """
        判断是否达到异步 compression 的 Soft Threshold。
        Soft Threshold 只负责提前触发 compression，不代表 Context 已经发生 hard overflow。
        """
        return self.total_tokens > self.soft_limit
    
    def is_overflow(self) -> bool:
        """
        判断是否真正超过 Hard Overflow Threshold。
        Hard Overflow 表示 Context 已经进入必须同步裁剪的危险区。
        """
        return self.total_tokens > self.hard_limit
        
    def get_messages_for_llm(self) -> List[Dict[str, Any]]:
        """
        返回当前消息列表。

        如果发生 Hard Overflow：
        1. system messages 始终保留；
        2. non-system messages 按 Message Block 划分；
        3. 从最近的 block 开始，在 hard token budget 内尽可能保留；
        4. Hard Trim 只能在 block 边界切分；
        5. 不允许拆开 assistant(tool_calls) + tool result(s)。

        裁剪会修改内部状态（messages 和 total_tokens）。
        """
        if not self.is_overflow():
            return self.messages.copy()

        # ========================================================
        # 1. system messages 始终保留
        # ========================================================
        system_msgs = [
            message
            for message in self.messages
            if message.get("role") == "system"
        ]
        non_system_msgs = [
            message
            for message in self.messages
            if message.get("role") != "system"
        ]

        target_tokens = self.hard_limit

        kept_tokens = sum(
            TokenEstimator.estimate_message(message)
            for message in system_msgs
        )

        # ========================================================
        # 2. non-system messages 按 Message Block 划分
        #
        # Hard Trim 与 Compression 共用同一个
        # Tool Calling protocol boundary invariant。
        # ========================================================
        blocks = _build_message_blocks(non_system_msgs)
        kept_blocks: List[List[Dict[str, Any]]] = []

        # ========================================================
        # 3. 从最近的 block 开始保留
        #
        # 注意：
        # budget 判断单位是整个 block，
        # 不再是单条 message。
        # ========================================================
        for block in reversed(blocks):
            block_tokens = sum(
                TokenEstimator.estimate_message(message)
                for message in block
            )
            if kept_tokens + block_tokens > target_tokens: 
                break

            kept_blocks.append(block)
            kept_tokens += block_tokens

        # ========================================================
        # 4. 恢复原始 block 顺序并 flatten
        # ========================================================
        kept_blocks.reverse()

        kept_non_system = _flatten_message_blocks(kept_blocks)

        # ========================================================
        # 5. 重建 Context，并重新计算真实 token
        # ========================================================
        self.messages = (system_msgs+ kept_non_system)

        self.recalculate_total_tokens()
        return self.messages.copy()
    
# ============================================================
# 4. 上下文管理器（核心调度器）
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
        if self.window.should_compress():
            self._trigger_async_compression()

    def add_assistant_message(self, content: str) -> None:
        self.window.add_message("assistant", content)
        # TODO: 接入 event_bus 记录此事件
        if self.window.should_compress():
            self._trigger_async_compression()

    def add_tool_result(self, tool_call_id: str, result: str) -> None:
        self.window.add_message_dict(
            {
                "role":"tool",
                "tool_call_id":tool_call_id,
                "content":result
            }
        )
        # TODO: 接入 event_bus 记录此事件
        if self.window.should_compress():
            self._trigger_async_compression()

    def add_system_message(self, content: str) -> None:
        """
        添加 system 消息（不会被裁剪）
        保持原来的“插入窗口头部”语义，但统一通过 ContextWindow 维护 token。
        
        """
        self.window.add_message_dict(
            {
                "role": "system",
                "content": content,
            },
            prepend=True,
        )
        # TODO: 接入 event_bus 记录此事件

    def _trigger_async_compression(self) -> None:
        """触发异步后台压缩（不阻塞）"""
        if self._background_task is None or self._background_task.done():
            self._background_task = asyncio.create_task(self._compress_background())
            # TODO: 接入 event_bus 记录压缩触发事件（包含当前 token 数、消息数量）

    def add_assistant_tool_calls(self,tool_calls):
        self.window.add_message_dict(
            {
                "role":"assistant",
                "content": None,
                "tool_calls":tool_calls
            }
        )
        if self.window.should_compress():
            self._trigger_async_compression()

    def _is_history_summary(
        self,
        message: Dict[str, Any],
    ) -> bool:
        """
        判断一条 system message 是否为 rolling historical summary。
        P0-3 当前采用最小实现：
        通过 [历史摘要] 前缀区分历史摘要与 pinned system instruction。

        注意：
        historical summary 虽然使用 role=system 发送给 LLM，
        但生命周期与真正的 pinned system instruction 不同：
        - pinned system: 长期保留
        - history summary: 每轮 compression 滚动替换
        """

        content = message.get("content")

        return (
            message.get("role") == "system"
            and isinstance(content, str)
            and content.startswith("[历史摘要]")
        )

    async def _compress_background(self) -> None:
        """
        后台压缩历史 Context。

        P0-2：
        1. 非 system messages 按 Message Block 划分；
        2. 最近 3 个 block 保留原文；
        3. 更早 blocks 进入 summary；
        4. compression boundary 不拆分
        assistant(tool_calls) + tool result(s)。

        P0-3：
        1. pinned system instruction 长期保留；
        2. historical summary 最多只保留一个；
        3. 后续 compression 使用：
            old summary
            +
            newly evicted raw history
            ->
            new summary
        4. old summary 被 new summary 替换，不累积。
        """

        print("DEBUG compress background START")
        async with self._compression_lock:
            # 防止短时间内多次触发
            if self._compression_triggered:
                return
            self._compression_triggered = True
            try:
                # ==================================================
                # 1. 记录 compression 开始
                # ==================================================
                if self.event_bus is not None:
                    self.event_bus.append(
                        Event(
                            type=EventType.LLM_REQUEST,
                            data={
                                "action": "compress_start",
                                "msg_count": len(
                                    self.window.messages
                                ),
                                "total_tokens": (
                                    self.window.total_tokens
                                ),
                            },
                        )
                    )

                # ==================================================
                # 2. Context 生命周期分区
                #
                # P0-3：
                #
                # system message 不再被视为同一种生命周期。
                #
                # 1. pinned_system_msgs
                #       真正的 system instruction，长期保留；
                #
                # 2. existing_history_summary
                #       上一轮 rolling summary，
                #       本轮参与新 summary 生成，但不会原样累积；
                #
                # 3. non_system_msgs
                #       普通对话历史，继续交给 P0-2 Message Block。
                # ==================================================
                pinned_system_msgs = [
                    message
                    for message in self.window.messages
                    if (
                        message.get("role") == "system"
                        and not self._is_history_summary(message)
                    )
                ]
                history_summary_msgs = [
                    message
                    for message in self.window.messages
                    if self._is_history_summary(message)
                ]
                non_system_msgs = [
                    message
                    for message in self.window.messages
                    if message.get("role") != "system"
                ]

                # P0-3 invariant：
                # 新代码正常运行时最多只会存在一个 historical summary。
                #
                # 如果当前窗口已有 summary，则取最新一条作为 rolling source。
                # 本层不扩展为通用 Context repair / Protocol Guard。
                existing_history_summary = (
                    history_summary_msgs[-1]
                    if history_summary_msgs
                    else None
                )

                if not non_system_msgs:
                    return

                # ==================================================
                # 3. 非 system messages 构造成 Message Blocks
                # ==================================================
                blocks = _build_message_blocks(non_system_msgs)

                #  保留最近 3 个 Message Block
                keep_recent_blocks = 3

                # 没有“旧历史”可以压缩时，不调用 summarizer，也不破坏近期原始消息。
                if len(blocks) <= keep_recent_blocks:
                    return

                history_blocks = (blocks[:-keep_recent_blocks])
                recent_blocks = (blocks[-keep_recent_blocks:])

                # ==================================================
                # 4. flatten：
                #
                # old history -> summary
                # recent       -> raw
                # ==================================================
                history_to_summarize = _flatten_message_blocks(history_blocks)
                kept_messages = _flatten_message_blocks(recent_blocks)
                
                if not history_to_summarize:
                    return

                # ==================================================
                # 5. 构造 rolling summary 输入
                #
                # P0-3：
                #
                # 第一次 compression：
                #     newly evicted raw history
                #           ↓
                #       new summary
                #
                # 后续 compression：
                #     old summary
                #     +
                #     newly evicted raw history
                #           ↓
                #       new summary
                #
                # recent raw messages 仍然不进入 summary，
                # 继续遵守 P0-2 old/recent partition。
                # ==================================================

                summary_source: List[Dict[str, Any]] = []
                
                if existing_history_summary is not None:
                    summary_source.append(existing_history_summary)

                summary_source.extend(history_to_summarize)
                summary_prompt = (
                    "请用一段话（不超过50字）"
                    "总结以下对话的核心内容："
                )
                summary_messages = [
                    {
                        "role": "system",
                        "content": summary_prompt,
                    },
                    {
                        "role": "user",
                        "content": str(summary_source),
                    },
                ]

                try:
                    # ==============================================
                    # 6. LLM summary request
                    # ==============================================
                    if self.event_bus is not None:
                        self.event_bus.append(
                            Event(
                                type=EventType.LLM_REQUEST,
                                data={"action": "llm_summarize"},
                            )
                        )
                    print(
                        "DEBUG compression messages:",
                        summary_messages,
                    )
                    response = await self.llm.generate(summary_messages)
                    print(
                        "DEBUG compression response:",
                        response,
                    )
                    if self.event_bus is not None:
                        self.event_bus.append(
                            Event(
                                type=EventType.SUCCESS,
                                data={
                                    "action": ("llm_summarize"),
                                    "summary": response.get("content","",),
                                },
                            )
                        )
                    summary = response.get(
                        "content",
                        "对话摘要生成失败",
                    )
                except Exception as e:

                    # ==============================================
                    # 7. 保留当前已有失败行为
                    #
                    # 本轮不扩大到 compression failure policy。
                    # ==============================================
                    if self.event_bus is not None:
                        self.event_bus.append(
                            Event(
                                type=EventType.ERROR,
                                data={
                                    "action": "compress",
                                    "exception": str(e),
                                },
                            )
                        )
                    summary = f"[压缩失败: {e}]"

                # ==================================================
                # 8. 更新当前 summary
                # 注意：
                # P0-3 才处理：
                # old summary + newly evicted history
                #                  ↓
                #             one new summary
                #
                # 本轮暂不提前修改 summary lifecycle。
                # ==================================================
                self.summary = summary

                # ==================================================
                # 9. 重建 Context
                # system
                # +
                # summary
                # +
                # recent raw blocks
                # ==================================================
                new_messages = pinned_system_msgs.copy()
                if self.summary:
                    new_messages.append(
                        {
                            "role": "system",
                            "content": (
                                f"[历史摘要] "
                                f"{self.summary}"
                            ),
                        }
                    )
                new_messages.extend(kept_messages)

                # ==================================================
                # 10. P0-1 invariant：
                # 完整重新计算 token。
                # ==================================================
                self.window.messages = new_messages
                self.window.recalculate_total_tokens()
                if self.event_bus is not None:
                    print("DEBUG compress DONE")
                    self.event_bus.append(
                        Event(
                            type=EventType.SUCCESS,
                            data={
                                "action": "compress_done",
                                "new_tokens": (
                                    self.window.total_tokens
                                ),
                            },
                        )
                    )
            finally:
                print("DEBUG compression FINALLY")
                self._compression_triggered = False

    def get_context_for_llm(self) -> List[Dict[str, str]]:
        """
        Runtime 调用此接口，自动判断是否裁剪上下文；
        获取最终发给 LLM 的消息列表（包含 summary + 最近消息）
        """
        # 同步裁剪（如果超限）
        return self.window.get_messages_for_llm()
    
# ============================================================
# 5. Prompt Builder
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


