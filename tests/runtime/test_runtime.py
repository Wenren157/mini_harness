import asyncio
from mini_harness.core.runtime import HarnessRuntime, ToolRegistry, LLMClient
from mini_harness.infra.config import RuntimeConfig
from mini_harness.core.models import Event, EventType

# ---------- Mock 实现 ----------
class MockLLMClient(LLMClient):
    def __init__(self, mode="final"):
        self.mode = mode  # "final", "tool", "multi"

    async def generate(self, messages, tools=None):
        await asyncio.sleep(0.1)
        if self.mode == "final":
            return {"content": "The answer is 42.", "tool_calls": []}
        elif self.mode == "tool":
            return {
                "content": None,
                "tool_calls": [
                    {"name": "add", "arguments": {"a": 1, "b": 2}, "id": "call_1"}
                ]
            }
        elif self.mode == "multi":
            return {
                "content": None,
                "tool_calls": [
                    {"name": "add", "arguments": {"a": 1, "b": 2}, "id": "call_1"},
                    {"name": "multiply", "arguments": {"a": 3, "b": 4}, "id": "call_2"}
                ]
            }
        else:
            return {"content": "Unknown", "tool_calls": []}

class MockToolRegistry(ToolRegistry):
    def __init__(self):
        self._tools = {}

    def register(self, name, func, description, parameters):
        self._tools[name] = {"func": func, "desc": description, "params": parameters}

    def get_tool(self, name):
        return self._tools.get(name)

    async def execute(self, name, **kwargs):
        await asyncio.sleep(1.0)  # 模拟耗时
        if name == "add":
            return kwargs.get("a", 0) + kwargs.get("b", 0)
        elif name == "multiply":
            return kwargs.get("a", 0) * kwargs.get("b", 0)
        else:
            return f"Executed {name} with {kwargs}"

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

    registry = MockToolRegistry()
    registry.register(
        "add", 
        None, 
        "Add two numbers", 
        {"a": "int", "b": "int"}
    )
    registry.register(
        "multiply", 
        None, 
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

if __name__ == "__main__":
    asyncio.run(main())