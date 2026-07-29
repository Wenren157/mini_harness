import os
from typing import List, Dict, Any, Optional
import openai

class BaseLLMClient:
    """LLM 客户端抽象基类"""
    async def generate(self, messages: List[Dict[str, str]]) -> Dict[str, str]:
        raise NotImplementedError
    
class OpenAILLMClient(BaseLLMClient):
    """OpenAI 兼容 API 客户端（支持 DeepSeek 等）"""
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.temperature = temperature
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment or passed to constructor")

        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    async def generate(self, messages: List[Dict[str, str]]) -> Dict[str, str]:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return {"content": response.choices[0].message.content}
    
class DeepSeekLLMClient(OpenAILLMClient):
    """DeepSeek API 客户端（直接使用 OpenAI 兼容接口）"""
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat", **kwargs):
        base_url = "https://api.deepseek.com/v1"
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)