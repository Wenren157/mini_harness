from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, List, Dict
import time

# ---------- 事件类型（Event Driven） ----------
class EventType(Enum):
    USER_INPUT = "user_input"          # 用户输入
    LLM_REQUEST = "llm_request"        # 请求LLM
    LLM_RESPONSE = "llm_response"      # LLM返回
    TOOL_CALL_REQUEST = "tool_call_request"   # 需要调用工具
    TOOL_CALL_RESULT = "tool_call_result"     # 工具返回结果
    AGENT_THINKING = "agent_thinking"  # Agent思考中（日志用）
    LOOP_ITERATION = "loop_iteration"  # 循环轮次
    ERROR = "error"                    # 异常
    FINISH = "finish"                  # 任务结束
    WARNING = "warning"                # 新增这行（用于记录重复告警，不影响主流程）
    SUCCESS = "success"                # 新增成功返回

@dataclass
class Event:
    type: EventType
    timestamp: float = field(default_factory=time.time)
    data: Any = None                  # 具体内容：str / dict / 工具调用结果
    trace_id: str = ""               # 链路追踪ID（Day 9 会用）

# ---------- Agent 状态机（State Machine） ----------
class AgentStatus(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    CALLING_TOOL = "calling_tool"
    EXECUTING = "executing"
    FINISHED = "finished"
    ERROR = "error"

@dataclass
class AgentState:
    status: AgentStatus = AgentStatus.IDLE
    # messages: List[Dict[str, str]] = field(default_factory=list)  # [{"role": "user/assistant/tool", "content": "..."}]
    current_iteration: int = 0
    max_iterations: int = 10
    pending_tool_calls: List[Dict] = field(default_factory=list)  # 待执行的工具列表
    final_answer: Optional[str] = None
    error_info: Optional[str] = None

    # ===== 新增：防死循环的“指纹记忆”（底软防重放机制） =====
    _last_tool_call_signature: Optional[str] = None      # 上一次工具调用的指纹
    _consecutive_duplicate_count: int = 0                # 连续重复次数