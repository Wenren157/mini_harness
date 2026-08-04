import asyncio
import sys
import os
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_harness.core.runtime import HarnessRuntime, RuntimeConfig
from mini_harness.infra.tools import create_default_tools
from tests.mocks import MockLLMClient


@pytest.mark.asyncio
async def test_integration(tmp_path):
    print("=" * 60)
    print("正确的集成测试：Runtime + Tools (调用 run 方法)")
    print("=" * 60)

    # 1. 创建真实组件
    workspace = str(tmp_path)
    tools = create_default_tools(workspace)
    # 准备一个会返回 tool_calls 的 Mock LLM
    llm = MockLLMClient(mode="multi")  # 让 LLM 一上来就要求调用工具
    config = RuntimeConfig(
        tool_timeout=3.0,
        max_iterations=2,  # 限制轮次，防止死循环跑太久
        enable_concurrent_tools=True
    )

    runtime = HarnessRuntime(
        llm_client=llm,
        tool_registry=tools,
        config=config
    )

    # 2. 执行公开接口 run()
    #    这里会走：run -> _step -> tools.execute -> 超时/重试 -> 记录 event
    print("\n>>> 执行 Runtime.run()，让内部循环自然调用 Tools...")
    
    # 注意：MockLLMClient 的 mode="multi" 返回的是 add/multiply，并不存在，会触发重试和错误
    # 为了测试超时，我们临时修改 MockLLMClient 的行为，或者直接使用自定义的 generate
    # 最干净的方式：给 MockLLMClient 加一个 set_response 方法，或者直接覆写
    async def custom_generate(messages):
        # 第一次调用返回 tool_calls（触发工具执行），第二次返回最终答案（结束循环）
        if len(messages) <= 2:  # 简单判断是否为第一轮
            return {
                "content": None,
                "tool_calls": [
                    {"name": "read_file", "arguments": {"path": "a.txt"}, "id": "call_1"},
                    {"name": "execute_command", "arguments": {"command": "sleep 10"}, "id": "call_2"},  # 触发超时
                    {"name": "read_file", "arguments": {"path": "a.txt"}, "id": "call_3"}
                ]
            }
        else:
            return {"content": "任务完成，但有些工具出错了。", "tool_calls": []}
    
    # 注入自定义响应逻辑
    llm.generate = custom_generate

    # 运行（这里会自动调用 _step，完全模拟真实场景）
    result = await runtime.run("请帮我处理这些文件。")

    # 添加下面这一行，看看工具到底返回了什么
    print("\n" + "=" * 60)
    print("State 完整消息列表（调试）:")
    print("=" * 60)
    for msg in runtime.context.get_context_for_llm():
        print(f"  {msg}")
    
    # 3. 输出结果
    print(f"\n最终答案: {result['final_answer']}")
    print(f"执行轮次: {result['iterations']}")
    print(f"最终状态: {result['status']}")

    print("\n" + "=" * 60)
    print("event_bus 中的 ERROR 事件:")
    print("=" * 60)
    errors = [evt for evt in runtime.event_bus if evt.type.value == "error"]
    
    if errors:
        print(f"✅ 验证通过！共捕获 {len(errors)} 条 ERROR 事件：")
        for evt in errors:
            print(f"  - {evt.data}")
    else:
        print("❌ 验证失败：未捕获到 ERROR 事件。")

    # 4. 检查状态机是否记录错误
    print("\n" + "=" * 60)
    print("State 错误信息:")
    print("=" * 60)
    print(f"错误状态: {runtime.state.error_info}")

if __name__ == "__main__":
    asyncio.run(test_integration())