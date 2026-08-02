import sys
import json
import asyncio
import os
from typing import Dict, Any, Optional

from .protocol import (
    JSONRPCRequest, JSONRPCResponse,
    make_success_response, make_error_response,
    MCPErrorCode
)
from mini_harness.infra.tools import create_default_tools
import mini_harness.infra.tools as tools


class MCPServer:
    def __init__(self, workspace: Optional[str] = None):
        

        self.registry = create_default_tools()

        self.tools = [
            {
                "name": "read_file",
                "description": "读取文件内容",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "文件路径"}
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "write_file",
                "description": "写入文件内容",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "文件路径"},
                        "content": {"type": "string", "description": "文件内容"}
                    },
                    "required": ["path", "content"]
                }
            },
            {
                "name": "execute_command",
                "description": "执行 Shell 命令",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell 命令"}
                    },
                    "required": ["command"]
                }
            }
        ]

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        if method == "initialize":
            return make_success_response(req_id, {"protocolVersion": "2024-11-05"})
        elif method == "tools/list":
            return make_success_response(req_id, {"tools": self.tools})
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            return await self._handle_tool_call(req_id, tool_name, arguments)
        else:
            return make_error_response(req_id, MCPErrorCode.METHOD_NOT_FOUND, f"Method not found: {method}")

    async def _handle_tool_call(self, req_id: Optional[int], tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        try:


            if tool_name == "read_file":
                result = await self.registry.execute_with_retry(
                            "read_file",
                            path=arguments.get("path", "")
                        )
                
                if isinstance(result, dict):
                    if result.get("error"):
                        raise Exception(result["error"])
                    content = result.get("content", "")
                    return make_success_response(req_id, {"content": content})
                else:
                    return make_success_response(req_id, {"content": result})

            elif tool_name == "write_file":
                result = await self.registry.execute_with_retry(
                            "write_file",
                            path=arguments.get("path",""),
                            content=arguments.get("content","")
                        )
                if isinstance(result, dict):
                    if result.get("error"):
                        raise Exception(result["error"])
                    # 有些实现返回 success 布尔值
                    if "success" in result and not result["success"]:
                        raise Exception("Write operation failed")
                    return make_success_response(req_id, {"content": "success"})
                else:
                    return make_success_response(req_id, {"content": "success"})

            elif tool_name == "execute_command":
                result = await self.registry.execute_with_retry(
                            "execute_command",
                            command=arguments.get("command","")
                        )
                if isinstance(result, dict):
                    if result.get("error"):
                        raise Exception(result["error"])
                    stdout = result.get("stdout", "")
                    return make_success_response(req_id, {"content": stdout})
                else:
                    return make_success_response(req_id, {"content": result})

            else:
                return make_error_response(req_id, MCPErrorCode.TOOL_NOT_FOUND, f"Tool not found: {tool_name}")
        except Exception as e:
            return make_error_response(req_id, MCPErrorCode.TOOL_EXECUTION_FAILED, str(e))

    async def run_stdio(self):
        sys.stderr.write("MCP Server started, waiting for requests...\n")
        sys.stderr.flush()
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = await self.handle_request(request)
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
            except json.JSONDecodeError:
                sys.stdout.write(json.dumps({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": MCPErrorCode.PARSE_ERROR, "message": "Parse error"}
                }) + "\n")
                sys.stdout.flush()
            except Exception as e:
                sys.stdout.write(json.dumps({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": MCPErrorCode.INTERNAL_ERROR, "message": str(e)}
                }) + "\n")
                sys.stdout.flush()


if __name__ == "__main__":
    # 从环境变量读取 workspace，便于测试时配置
    workspace = os.environ.get("MCP_WORKSPACE")
    server = MCPServer(workspace=workspace) if workspace else MCPServer()
    asyncio.run(server.run_stdio())