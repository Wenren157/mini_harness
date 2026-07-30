from mini_harness.core.runtime import HarnessRuntime
from mini_harness.infra.config import RuntimeConfig
from mini_harness.infra.tools import ToolRegistry, SandboxExecutor
from tests.mocks import MockLLMClient


def test_context_injection():
    """
    检测 Context是否正确注入Runtime
    """
    config = RuntimeConfig()

    llm = MockLLMClient()

    sandbox = SandboxExecutor()

    registry = ToolRegistry(
        sandbox=sandbox
    )


    runtime = HarnessRuntime(
        config=config,
        llm_client=llm,
        tool_registry=registry
    )


    # 1. Context存在
    assert runtime.context is not None


    # 2. EventBus绑定正确
    assert runtime.context.event_bus is runtime.event_bus


    print("Context injection OK")