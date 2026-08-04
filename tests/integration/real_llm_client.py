"""
Real LLM Client 适配器
对接 DeepSeek API (OpenAI 兼容格式)，实现与 MockLLMClient 相同接口
"""

import os
import logging
from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI

from mini_harness.infra.llm_client import LLMClient  # 假设已有基类定义 generate() 签名

logger = logging.getLogger(__name__)


class RealLLMClient(LLMClient):
    """真实 LLM 适配器，通过依赖注入替换 MockLLMClient"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: int = 30,
    ):
        self.model = model
        # API Key 优先级：参数 > 环境变量
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY 未设置。请在环境变量或参数中提供。"
            )
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=timeout,
        )

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        调用 DeepSeek Chat API，返回统一格式的结果。

        Args:
            messages: 对话消息列表，标准 OpenAI 格式
            tools: 可选工具定义列表

        Returns:
            {
                "content": str | None,
                "tool_calls": [{"name": str, "arguments": dict}] | None,
                "finish_reason": str,
                "usage": {"prompt_tokens": int, "completion_tokens": int}
            }
        """
        try:
            # 构造请求参数
            kwargs = {
                "model": self.model,
                "messages": messages,
            }
            if tools:
                kwargs["tools"] = tools

            response = await self.client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            msg = choice.message

            # 统一输出格式
            result = {
                "content": msg.content,
                "tool_calls": None,
                "finish_reason": choice.finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                } if response.usage else None,
            }

            # 处理 tool_calls
            if msg.tool_calls:
                result["tool_calls"] = [
                    {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,  # 注意：这里是 JSON 字符串，与协议一致
                    }
                    for tc in msg.tool_calls
                ]

            return result

        except Exception as e:
            logger.error(f"RealLLMClient generate error: {e}")
            raise