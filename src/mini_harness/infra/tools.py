import asyncio
import os
from pathlib import Path
import sys
import subprocess
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass
import aiofiles
import time

MAX_READ_SIZE = 1 * 1024 * 1024      # 1 MB
MAX_WRITE_SIZE = 1 * 1024 * 1024     # 1 MB


# ---------- 工具定义（Tool Definition） ----------
@dataclass
class Tool:
    """单个工具的元数据与执行函数"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema 格式
    func: Callable[..., Awaitable[Any]]

# ---------- 沙箱执行器接口（Sandbox Executor） ----------
class SandboxExecutor:
    """
    负责在受限环境下执行工具。
    所有执行必须有超时、资源限制、错误捕获。
    """
    def __init__(
            self, 
            workspace_root="./workspace"
    ):
        self.workspace_root = Path(workspace_root)

        if not workspace_root:
            raise ValueError(
                "workspace_root is required"
            )
        self.workspace_root = os.path.abspath(workspace_root)
        os.makedirs(
            self.workspace_root,
            exist_ok=True
        )
    
    async def execute_shell(
            self, 
            command: str, 
            timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        执行 Shell 命令。
        返回: {"stdout": str, "stderr": str, "return_code": int, "execution_time": float}
        约束: 使用 asyncio.create_subprocess_shell + asyncio.wait_for 实现超时,超时或命令返回非0时，抛出异常
        """
        start = time.perf_counter()
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            # returncode = proc.returncode
            # timed_out = False
        except asyncio.TimeoutError:
            # 超时：强制终止进程，并读取剩余输出
            proc.kill()
            # if proc.stdout:
            #     proc.stdout.close()
            # if proc.stderr:
            #     proc.stderr.close()
            await proc.wait()  # 修复：等待进程结束，防止僵尸
            # ===== 核心修复：显式抛出异常，而不是返回 timed_out=True =====
            raise TimeoutError(f"命令 '{command}' 执行超时（限制 {timeout}s）")
            # stdout, stderr = await proc.communicate()  # 等待进程结束
            # returncode = proc.returncode
            # timed_out = True

        # 解码输出：根据操作系统选择正确的编码，避免中文乱码
        encoding = 'gbk' if sys.platform == 'win32' else 'utf-8'
        stdout_text = stdout.decode(encoding, errors='replace')
        stderr_text = stderr.decode(encoding, errors='replace')

        # ===== 额外加固：如果命令返回非0，也视为异常 =====
        if proc.returncode != 0:
            raise RuntimeError(f"命令 '{command}' 执行失败 \
                               (code {proc.returncode}): {stderr_text}")
    
        # execution_time = time.perf_counter() - start
        return {
            # "stdout": stdout.decode(encoding, errors='replace'),
            # "stderr": stderr.decode(encoding, errors='replace'),
            # "return_code": returncode,
            # "execution_time": execution_time,
            # "timed_out": timed_out,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "return_code": proc.returncode,
            "execution_time": time.perf_counter() - start,
        }
        
    
    async def read_file(self, path: str, timeout: float = 10.0) -> Dict[str, Any]:
        """
        异步读取文件，限制最大 1MB。
        返回: {"content": str, "size": int, "error": str or None}
        约束: 必须限制读取大小（例如最大 1MB），防止恶意读超大文件拖垮内存。
        若文件大小超过限制，直接返回错误，防止内存耗尽。
        """
        # 路径安全检查（与 write_file 共用）
        safe_path = self._safe_path(path)
        if safe_path is None:
            return {"content": "", "size": 0, "error": "路径越权或不存在"}
        
        try:
            # 先获取文件大小（同步调用，但使用 to_thread 避免阻塞）
            size = await asyncio.to_thread(os.path.getsize, safe_path)
            if size > MAX_READ_SIZE:
                return {
                    "content": "",
                    "size": size,
                    "error": f"文件大小 {size} 超过最大读取限制 {MAX_READ_SIZE}",
                }
            # 读取全部内容
            async with aiofiles.open(safe_path, mode="rb") as f:
                content = await f.read()
            return {
                "content": content.decode("utf-8", errors="replace"),
                "size": len(content),
                "error": None,
            }
        except FileNotFoundError:
            return {"content": "", "size": 0, "error": "文件不存在"}
        except Exception as e:
            return {"content": "", "size": 0, "error": str(e)}
        
    
    async def write_file(
            self, 
            path: str, 
            content: str, 
            timeout: float = 10.0
    ) -> Dict[str, Any]:
        """
        写入文件。
        返回: {"success": bool, "bytes_written": int, "error": str or None}
        约束: 必须限制写入大小，且路径必须限制在当前 Workspace 目录下（防止越权）。
        """
        safe_path = self._safe_path(path)

        print(
            "DEBUG write_file:",
            self.workspace_root,
            path,
            safe_path,
            file=sys.stderr
        )

        if safe_path is None:
            return {"success": False, "bytes_written": 0, "error": "路径越权"}
        
        if len(content) > MAX_WRITE_SIZE:
            return {
                "success": False,
                "bytes_written": 0,
                "error": f"写入内容大小 {len(content)} 超过最大限制 {MAX_WRITE_SIZE}",
            }
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(safe_path), exist_ok=True)
            async with aiofiles.open(safe_path, mode="w", encoding="utf-8") as f:
                bytes_written = await f.write(content)
            return {"success": True, "bytes_written": bytes_written, "error": None}
        except Exception as e:
            return {"success": False, "bytes_written": 0, "error": str(e)}
  
    def _safe_path(self, path: str) -> Optional[str]:
        """
        将传入路径规范化为绝对路径，并检查是否在 WORKSPACE_ROOT 内。
        防止目录遍历攻击（如 ../../../etc/passwd）。
        """
        abs_path = os.path.abspath(
            os.path.join(
                self.workspace_root,
                path
            )
        )
        real_path = os.path.realpath(abs_path)
        workspace_real = os.path.realpath(self.workspace_root)
        if os.path.commonpath(
            [
                real_path,
                workspace_real
            ]
        ) != workspace_real:
            return None
        
        return real_path
     

# ---------- 工具注册表实现（Tool Registry） ----------
class ToolRegistry:
    """
    管理所有可用工具。
    职责：注册、获取、执行（带重试机制）。
    强化点：执行必须支持“指数退避重试”（Retry with Exponential Backoff）。
    """
    
    def __init__(self, sandbox: SandboxExecutor):
        self.sandbox = sandbox
        self._tools: Dict[str, Tool] = {}
        self._max_retries: int = 3
        self._retry_base_delay: float = 0.5
    
    def register(
            self, 
            name: str, 
            func: Callable[..., Awaitable[Any]],
            description: str, 
            parameters: Dict[str, Any]
    ) -> None:
        
        """注册工具，参数描述遵循 OpenAI Function Calling 格式"""
        self._tools[name] = Tool(name, description, parameters, func)
    
    def get_schema(self) -> List[Dict[str, Any]]:
        """返回所有工具的 OpenAPI Schema 列表，用于发给 LLM"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters
                }
            }
            for t in self._tools.values()
        ]
    
    async def execute_with_retry(self, name: str, **kwargs) -> Any:
        """
        异步执行工具，失败时自动重试（指数退避）。
        逻辑：尝试 -> 失败 -> 等待 0.5s -> 尝试 -> 失败 -> 等待 1s -> 尝试 -> 失败 -> 抛异常。
        """
        if name not in self._tools:
            raise KeyError(f"工具 '{name}' 未注册")
        
        tool = self._tools[name]
        last_exception = None

        for attempt in range(1, self._max_retries + 1):
            try:
                return await tool.func(**kwargs)
            except Exception as e:
                last_exception = e
                if attempt == self._max_retries:
                    break
                delay = self._retry_base_delay * (2 ** (attempt - 1))
                await asyncio.sleep(delay)
        raise RuntimeError(
            f"工具 '{name}' 执行失败，重试 {self._max_retries} 次后仍失败") \
            from last_exception

# ---------- 默认工具工厂（Factory） ----------
def create_default_tools(workspace: str) -> ToolRegistry:
    """
    创建包含以下默认工具的 Registry：
    1. execute_command   -> 执行 Shell 命令
    2. read_file         -> 读取文件
    3. write_file        -> 写入文件
    """
    sandbox = SandboxExecutor(workspace_root=workspace)
    registry = ToolRegistry(sandbox)
    # ... 注册逻辑 ...

    # 工具1：execute_command
    registry.register(
        name="execute_command",
        description="在沙箱中执行 Shell 命令",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的命令"},
                "timeout": {"type": "number", "description": "超时秒数（默认30）", "default": 30.0},
            },
            "required": ["command"],
        },
        func=lambda command, timeout=30.0: sandbox.execute_shell(command, timeout)
    )

    # 工具2：read_file
    registry.register(
        name="read_file",
        description="读取沙箱内的文件（最大1MB）",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件相对路径（相对于workspace）"},
                "timeout": {"type": "number", "description": "超时秒数（默认10）", "default": 10.0},
            },
            "required": ["path"],
        },
        func=lambda path, timeout=10.0: sandbox.read_file(path, timeout)
    )

    # 工具3：write_file
    registry.register(
        name="write_file",
        description="写入文件到沙箱内（最大1MB）",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件相对路径（相对于workspace）"},
                "content": {"type": "string", "description": "文件内容"},
                "timeout": {"type": "number", "description": "超时秒数（默认10）", "default": 10.0},
            },
            "required": ["path", "content"],
        },
        func=lambda path, content, timeout=10.0: sandbox.write_file(path, content, timeout)
    )
    
    return registry