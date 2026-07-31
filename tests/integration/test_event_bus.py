import pytest
from mini_harness.core.runtime import HarnessRuntime, RuntimeConfig
from mini_harness.infra.tools import ToolRegistry
from mini_harness.core.models import EventType

class MockLLMClient:
    async def generate(self, messages):
        return {"content": "Mock response", "tool_calls": []}

@pytest.mark.asyncio
async def test_event_bus_truncation():
    config = RuntimeConfig(max_iterations=5, event_bus_maxlen=5)
    llm = MockLLMClient()
    tools = ToolRegistry()
    runtime = HarnessRuntime(config, llm, tools)

    # 添加 10 个事件
    for i in range(10):
        runtime._record_event(EventType.LOOP_ITERATION, i)

    assert len(runtime.event_bus) == 5
    # 检查最早保留的事件是索引 5
    assert runtime.event_bus[0].data == 5
    assert runtime.event_bus[-1].data == 9