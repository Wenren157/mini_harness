import asyncio
import json
import logging
import os
import sys
import tempfile

import pytest
import pytest_asyncio

from mini_harness.mcp.client import MCPClient
from mini_harness.mcp.server import MCPServer


class FakeMCPStdin:
    """记录 Client 写入的 JSON-RPC 请求。"""

    def __init__(self):
        self.data = b""

    def write(self, data):
        self.data += data

    async def drain(self):
        return None


class FakeMCPStdout:
    """向 Client 返回一条预设 JSON-RPC 响应。"""

    def __init__(self, response):
        self.response_line = (
            json.dumps(response).encode("utf-8")
            + b"\n"
        )

    async def readline(self):
        response_line = self.response_line
        self.response_line = b""
        return response_line


class FakeMCPStderr:
    async def read(self):
        return b""


class FakeMCPProcess:
    def __init__(self, response):
        self.stdin = FakeMCPStdin()
        self.stdout = FakeMCPStdout(response)
        self.stderr = FakeMCPStderr()


@pytest.mark.asyncio
async def test_client_logs_metadata_without_exposing_payload(
    capsys,
    caplog,
):
    """Client 应记录调用元数据，但不输出或记录工具结果。"""
    sensitive_path = "private/source.py"
    sensitive_content = "SECRET_SOURCE_CONTENT"

    process = FakeMCPProcess(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": sensitive_content,
            },
        }
    )
    client = MCPClient(process)

    with caplog.at_level(
        logging.DEBUG,
        logger="mini_harness.mcp.client",
    ):
        result = await client.call_tool(
            "read_file",
            {
                "path": sensitive_path,
            },
        )

    captured = capsys.readouterr()

    assert result == sensitive_content

    assert captured.out == ""
    assert captured.err == ""

    assert "tools/call" in caplog.text
    assert "read_file" in caplog.text

    assert sensitive_path not in caplog.text
    assert sensitive_content not in caplog.text


@pytest.mark.asyncio
async def test_server_logs_metadata_without_exposing_arguments(
    tmp_path,
    capsys,
    caplog,
):
    """Server 应记录方法和工具名，但不记录参数值。"""
    sensitive_content = "SECRET_WRITE_CONTENT"

    server = MCPServer(
        workspace=str(tmp_path)
    )

    request = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {
            "name": "write_file",
            "arguments": {
                "path": "output.txt",
                "content": sensitive_content,
            },
        },
    }

    with caplog.at_level(
        logging.DEBUG,
        logger="mini_harness.mcp.server",
    ):
        response = await server.handle_request(request)

    captured = capsys.readouterr()

    assert "result" in response
    assert response["result"]["content"] == "success"

    assert captured.out == ""
    assert captured.err == ""

    assert "tools/call" in caplog.text
    assert "write_file" in caplog.text

    assert sensitive_content not in caplog.text


# 确保项目 src 目录在 PYTHONPATH 中（子进程需要）
# 在测试 fixture 中动态设置


@pytest_asyncio.fixture
async def mcp_client():
    # 项目根目录：假设测试文件在 tests/mcp/ 下，向上三级到达项目根
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    src_dir = os.path.join(project_root, "src")

    # 构建子进程的环境变量，加入 src 目录到 PYTHONPATH
    env = os.environ.copy()
    pythonpath = src_dir
    if "PYTHONPATH" in env:
        pythonpath += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    env["MCP_WORKSPACE"] = project_root

    # 启动 MCP Server（通过 -m 方式）
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "mini_harness.mcp.server",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=project_root,
        env=env,
    )

    # 启动一个后台任务打印 stderr（便于调试）
    async def read_stderr():
        while True:
            line = await process.stderr.readline()
            if not line:
                break
            print(
                f"[MCP Server stderr] "
                f"{line.decode('utf-8', errors='replace').strip()}"
            )

    asyncio.create_task(read_stderr())

    client = MCPClient(process)
    yield client
    await client.close()


@pytest.mark.asyncio
async def test_initialize(mcp_client):
    resp = await mcp_client.initialize()
    assert resp["result"]["protocolVersion"] == "2024-11-05"


@pytest.mark.asyncio
async def test_list_tools(mcp_client):
    tools = await mcp_client.list_tools()
    assert len(tools) == 3
    names = {t["name"] for t in tools}
    assert names == {"read_file", "write_file", "execute_command"}


@pytest.mark.asyncio
async def test_call_read_file(mcp_client):
    with tempfile.NamedTemporaryFile(
        mode="w+",
        delete=False,
        dir="workspace"
    ) as f:
        f.write("Hello, MCP!")
        tmp_path = f.name
    try:
        content = await mcp_client.call_tool("read_file", {"path": tmp_path})
        assert content == "Hello, MCP!"
    finally:
        os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_call_write_file(mcp_client):
    with tempfile.NamedTemporaryFile(
        mode="w+",
        delete=False,
        dir="workspace"
    ) as f:
        tmp_path = f.name
    try:
        await mcp_client.call_tool("write_file", {"path": tmp_path, "content": "New content"})
        with open(tmp_path, "r") as f:
            assert f.read() == "New content"
    finally:
        os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_call_execute_command(mcp_client):
    result = await mcp_client.call_tool("execute_command", {"command": "echo hello"})
    assert "hello" in result


@pytest.mark.asyncio
async def test_call_unknown_tool(mcp_client):
    resp = await mcp_client._send_request("tools/call", {"name": "unknown", "arguments": {}})
    assert "error" in resp
    assert resp["error"]["code"] == -32001  # TOOL_NOT_FOUND