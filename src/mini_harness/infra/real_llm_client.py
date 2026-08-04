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

        # 调试打印：
        print(
            "\n========== DEBUG LLM REQUEST ==========",
            flush=True
        )

        print(
            "TOOLS:",
            tools,
            flush=True
        )

        print(
            "MESSAGES:",
            json.dumps(
                messages,
                indent=2,
                ensure_ascii=False,
                default=str
            ),
            flush=True
        )

        print(
            "=======================================\n",
            flush=True
        )

        if tools:
            kwargs["tools"] = tools

        try:
            response = await self.client.chat.completions.create(
                **kwargs
            )
        except Exception as e:
            print(
                "\n========== DEBUG LLM ERROR ==========",
                flush=True
            )

            print(
                type(e).__name__,
                str(e),
                flush=True
            )

            print(
                "=====================================\n",
                flush=True
            )

            raise
        choice = response.choices[0]
        message = choice.message

        # 调试打印
        print(
            "\n========== DEBUG LLM RESPONSE ==========",
            flush=True
        )

        print(
            "CONTENT:",
            message.content,
            flush=True
        )

        print(
            "TOOL_CALLS:",
            message.tool_calls,
            flush=True
        )

        print(
            "========================================\n",
            flush=True
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