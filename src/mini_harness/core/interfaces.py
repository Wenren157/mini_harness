from typing import List, Dict, Optional, Any

class LLMClient:
    """大模型调用接口（Runtime 依赖契约）"""
    async def generate(
        self, 
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        调用LLM，返回响应。
        返回格式必须包含：
        {
            "content": "文本回复" 或 None（如果有工具调用）,
            "tool_calls": [{"name": "read_file", "arguments": {"path": "/tmp/a.txt"}}, ...] 或 []
        }
        """
        raise NotImplementedError