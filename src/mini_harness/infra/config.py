from dataclasses import dataclass
import os


@dataclass
class RuntimeConfig:
    llm_model: str = "gpt-4o-mini"          # 或你本地部署的Qwen
    api_base: str = "http://localhost:8000/v1"
    api_key: str = "sk-xxx"
    tool_timeout: float = 30.0              # 单个工具最长执行时间（秒）
    max_iterations: int = 10
    enable_concurrent_tools: bool = True    # 是否开启并发工具调用
    max_context_tokens: int = 8000          # <--- 新增这一行
    event_bus_maxlen: int = 10000           # 新增
    enable_mcp: bool = False
    workspace: str = "./workspace"   # 唯一workspace入口

    def __post_init__(self):
        """
        Runtime资源初始化
        """
        self.workspace = os.path.abspath(self.workspace)
        os.makedirs(
            self.workspace,
            exist_ok=True
        )