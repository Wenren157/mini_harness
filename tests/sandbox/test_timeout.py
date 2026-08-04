# tests/sandbox/test_timeout.py
import asyncio
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mini_harness.infra.tools import SandboxExecutor

@pytest.mark.asyncio
async def test_timeout():
    sandbox = SandboxExecutor()
    
    print("=== 测试 execute_shell 超时 ===")
    # Windows: timeout /t 60，Linux: sleep 60
    cmd = "timeout /t 60" if sys.platform == "win32" else "sleep 60"
    try:
        result = await sandbox.execute_shell(cmd, timeout=3.0)
        print(f"返回结果（未超时？）: {result}")
    except asyncio.TimeoutError:
        print("✅ 超时生效！")
    except Exception as e:
        print(f"❌ 其他异常: {e}")

if __name__ == "__main__":
    asyncio.run(test_timeout())