import asyncio
import os
import sys
import tempfile
import pytest
import pytest_asyncio

from mini_harness.mcp.client import MCPClient

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