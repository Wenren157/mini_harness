
import asyncio
from typing import List, Dict, Any, Optional
from mini_harness.core.runtime import LLMClient

# ---------- Mock 实现 ----------
class MockLLMClient(LLMClient):
    """
    模拟 LLM 客户端，用于单元测试和集成测试。
    支持预设不同的返回模式：final（纯文本）、tool（单工具调用）、multi（多工具调用）。
    """
    def __init__(self, mode="final"):
        """
        Args:
            mode: "final" | "tool" | "multi"
                - "final": 返回最终答案
                - "tool": 返回单个工具调用
                - "multi": 返回多个工具调用
        """
        self.mode = mode  # "final", "tool", "multi"
        self.call_history: List[Dict] = []  # 记录每次调用，便于断言

    async def generate(
            self, 
            messages: List[Dict[str, str]], 
            tools: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """模拟 LLM 生成，记录调用历史并返回预设值"""
        self.call_history.append({
            "messages": messages.copy(),
            "tools": tools,
            "timestamp": asyncio.get_event_loop().time()
        })

        await asyncio.sleep(0.1) # 模拟网络延迟

        if self.mode == "final":
            return {"content": "The answer is 42.", "tool_calls": []}
        elif self.mode == "tool":
            return {
                "content": None,
                "tool_calls": [
                    {"name": "add", "arguments": {"a": 1, "b": 2}, "id": "call_1"}
                ]
            }
        elif self.mode == "multi":
            return {
                "content": None,
                "tool_calls": [
                    {"name": "add", "arguments": {"a": 1, "b": 2}, "id": "call_1"},
                    {"name": "multiply", "arguments": {"a": 3, "b": 4}, "id": "call_2"}
                ]
            }
        else:
            return {"content": "Unknown", "tool_calls": []}
    
    def set_mode(self, mode: str):
        """动态切换模式，方便在测试中灵活调整"""
        self.mode = mode
    
    def reset_history(self):
        """清空调用历史，便于每个测试独立验证"""
        self.call_history = []