"""
Real LLM Client
DeepSeek(OpenAI Compatible) Adapter
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI
from mini_harness.core.interfaces import LLMClient

logger = logging.getLogger(__name__)


class RealLLMClient(LLMClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: int = 30,
    ):
        self.api_key = (
            api_key
            or os.getenv("DEEPSEEK_API_KEY")
        )

        if not self.api_key:
            raise ValueError(
                "Missing DEEPSEEK_API_KEY"
            )


        self.model = model
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=timeout,
        )

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools=None,
    ) -> Dict[str, Any]:
        
        kwargs = {
            "model": self.model,
            "messages": messages,
        }

        # 只记录结构元数据，不记录请求原文。
        logger.debug(
            (
                "LLM request started: "
                "model=%s message_count=%s tool_count=%s"
            ),
            self.model,
            len(messages),
            len(tools) if tools else 0,
        )

        if tools:
            kwargs["tools"] = tools

        try:
            response = await self.client.chat.completions.create(
                **kwargs
            )
        except Exception:
            logger.exception(
                "LLM request failed: model=%s",
                self.model,
            )
            raise

        choice = response.choices[0]
        message = choice.message

        logger.debug(
            (
                "LLM response received: "
                "model=%s finish_reason=%s "
                "has_content=%s tool_call_count=%s"
            ),
            self.model,
            choice.finish_reason,
            message.content is not None,
            len(message.tool_calls or []),
        )

        result = {
            "content": message.content,
            "tool_calls": [],
            "finish_reason": choice.finish_reason,
            "usage": None,
        }
        if response.usage:
            result["usage"] = {
                "prompt_tokens":
                    response.usage.prompt_tokens,
                "completion_tokens":
                    response.usage.completion_tokens,
            }
        if message.tool_calls:
            for call in message.tool_calls:
                arguments = call.function.arguments
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                result["tool_calls"].append(
                    {
                        "id": call.id,
                        "name": call.function.name,
                        "arguments": arguments
                    }
                )
        return result