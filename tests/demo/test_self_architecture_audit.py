"""
单元测试：验证 Demo 的只读约束和基本执行流程。
"""

import asyncio
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 从 demo 模块导入需要测试的函数
from demo.self_architecture_audit import (
    create_readonly_tool_registry,
    list_directory,
)


@pytest.mark.asyncio
async def test_readonly_tools():
    """确认只读工具注册表不包含写操作或命令执行工具"""
    registry = create_readonly_tool_registry(".")
    schema = registry.get_schema()
    tool_names = [s["function"]["name"] for s in schema]
    assert "read_file" in tool_names
    assert "list_directory" in tool_names
    assert "write_file" not in tool_names
    assert "execute_command" not in tool_names


@pytest.mark.asyncio
async def test_list_directory_works():
    """测试 list_directory 工具能正常工作"""
    registry = create_readonly_tool_registry(".")
    sandbox = registry.sandbox
    result = await list_directory(sandbox, ".")
    assert "tree" in result
    assert result.get("error") is None
    # 确保返回的树结构至少包含一些条目（非空）
    assert len(result["tree"]) > 0