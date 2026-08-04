import sys
import os
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from mini_harness.core.runtime import HarnessRuntime, LLMClient
from mini_harness.infra.tools import  ToolRegistry, SandboxExecutor  
from mini_harness.infra.config import RuntimeConfig
from mini_harness.core.models import Event, EventType
from tests.mocks import MockLLMClient  # ← 只导入 MockLLMClient
import tempfile

# ---------- 测试函数 ----------
async def run_test(
        mode: str, 
        query: str, 
        description: str
    ):

    config = RuntimeConfig(
        tool_timeout=5.0, 
        max_iterations=3, 
        enable_concurrent_tools=True,
        max_context_tokens = 50
    )

    # ========== 修复点：创建真实的 SandboxExecutor ==========
    workspace = tempfile.mkdtemp()
    sandbox = SandboxExecutor(workspace_root=workspace)  # 真实沙箱
    registry = ToolRegistry(sandbox=sandbox) # 传入 sandbox

    # 注册测试工具（注意：这里的 func 要传入实际可执行的异步函数）
    # 由于是测试，我们可以用简单的 lambda 或直接传递 None（但真实执行会报错）
    # 对于 mock 模式，建议使用真实的工具函数（如 add/multiply 的实现）
    async def add_func(a, b):
        return a + b
    
    async def multiply_func(a, b):
        return a * b
    
    registry.register(
        "add", 
        add_func,  # ← 需要传真实的异步函数
        "Add two numbers", 
        {"a": "int", "b": "int"}
    )
    registry.register(
        "multiply", 
        multiply_func,  # ← 同上
        "Multiply two numbers", 
        {"a": "int", "b": "int"}
    )

    llm = MockLLMClient(mode=mode)
    runtime = HarnessRuntime(
        config=config, 
        llm_client=llm, 
        tool_registry=registry
    )

    result = await runtime.run(query)
    print(f"\n=== {description} ===")
    print(f"Final answer: {result['final_answer']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Status: {result['status']}")
    print("Events:")
    for ev in result['events']:
        print(f"  {ev.type.value}: {ev.data}")

async def main():
    await run_test(
        "tool", 
        "What is 1+2?", 
        "Single tool call"
    )
    await run_test(
        "multi", 
        "Compute 1+2 and 3*4", 
        "Multi tool concurrent"
    )
    await run_test(
        "final", 
        "Just tell me the answer", 
        "Direct text response"
    )

# ========== 新增：独立测试函数 ==========
@pytest.mark.asyncio
async def test_runtime_final():
    """
    冒烟测试： 测试基础链路
    LLM 直接返回最终答案（不请求调用任何工具）时，Runtime 
    能否正确地结束循环并把答案交出来
    """
    print("\n" + "=" * 50)
    print("独立测试：Runtime 返回最终答案")
    print("=" * 50)

    llm = MockLLMClient(mode="final")

    workspace = tempfile.mkdtemp()
    sandbox = SandboxExecutor(workspace_root=workspace)
    # 真实 Registry，但不用注册工具，因为 LLM 不会调用工具
    registry = ToolRegistry(
        sandbox=sandbox
    )        
               
    config = RuntimeConfig(max_iterations=3)

    runtime = HarnessRuntime(
        config=config,
        llm_client=llm,
        tool_registry=registry
    )

    result = await runtime.run("What is the answer?")
    
    print(f"Final answer: {result['final_answer']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Status: {result['status']}")
    print("Events:")
    for ev in result['events']:
        print(f"  {ev.type.value}: {ev.data}")
    
    # 断言验证
    assert result["final_answer"] == "The answer is 42."
    print("\n✅ 测试通过！")

# ========== 入口 ==========
if __name__ == "__main__":
    # 如果你只想跑冒烟测试，取消下面一行的注释，注释掉 main()
    # asyncio.run(test_runtime_final())
    
    # 默认跑全量测试
    asyncio.run(main())