import asyncio
import json
import os
import sys
from typing import Dict, Any, List, Optional


class MCPClient:
    """
    MCP Client（stdin/stdout 通信）
    用于与 MCP Server 交互，调用远程工具。
    """

    def __init__(self, server_process):
        self.process = server_process
        self._request_id = 0
        # TODO: 未来支持多 Server 连接池管理

    @classmethod
    async def start(
        cls,
        module_name: str = "mini_harness.mcp.server",
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        """
        通过 `python -m module_name` 启动 MCP Server 子进程。
        推荐使用此方式启动，避免路径问题。
        """
        if env is None:
            env = os.environ.copy()
        # 如果未显式设置 cwd，则使用当前工作目录
        if cwd is None:
            cwd = os.getcwd()
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            module_name,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
        return cls(process)

    async def _send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """发送 JSON-RPC 请求并等待响应"""
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or {},
        }
        # 写入 stdin
        self.process.stdin.write(json.dumps(request).encode() + b"\n")
        await self.process.stdin.drain()
        # 读取 stdout（一行一个响应）
        response_line = await self.process.stdout.readline()
        if not response_line:
            # 尝试读取 stderr 获取错误信息
            stderr_data = await self.process.stderr.read()
            error_msg = stderr_data.decode().strip() if stderr_data else "No stderr output"
            raise RuntimeError(f"MCP Server closed connection. stderr: {error_msg}")
        return json.loads(response_line)

    async def initialize(self) -> Dict[str, Any]:
        return await self._send_request("initialize", {})

    async def list_tools(self) -> List[Dict[str, Any]]:
        response = await self._send_request("tools/list", {})
        return response.get("result", {}).get("tools", [])

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        response = await self._send_request(
            "tools/call", 
            {
                "name": name, 
                "arguments": arguments
            }
        )
        print("DEBUG MCP RESPONSE:", response)

        return response.get("result", {}).get("content", {})

    # TODO: 未来支持 resources/read 和 prompts/get

    async def close(self):
        self.process.terminate()
        await self.process.wait()