import unittest

from mini_harness.infra.memory import MemoryManager
from mini_harness.infra.context import ContextManager


# ============================================================
# Mock LLM
# ============================================================

class MockLLMClient:

    async def generate(self, messages):
        return {
            "content": "mock summary"
        }


# ============================================================
# Real ContextManager Integration Test
# ============================================================

class TestMemoryRealContextIntegration(unittest.TestCase):

    def test_memory_inject_into_real_context(self):

        # ==================================
        # Step1:
        # 创建 Memory
        # ==================================

        memory_manager = MemoryManager(
            max_size=10
        )

        memory_manager.add(
            "language",
            "Python"
        )


        # ==================================
        # Step2:
        # 获取 Memory Injection
        # ==================================

        injections = (
            memory_manager
            .get_context_injections()
        )


        self.assertEqual(
            len(injections),
            1
        )


        self.assertIn(
            "Python",
            injections[0]
        )


        # ==================================
        # Step3:
        # 创建真实 ContextManager
        # ==================================

        context_manager = ContextManager(
            llm_client=MockLLMClient()
        )


        # ==================================
        # Step4:
        # 将 Memory 注入 system message
        #
        # 对应真实运行逻辑：
        # Memory -> Context -> Prompt
        # ==================================

        for memory in injections:
            context_manager.add_system_message(
                memory
            )


        # ==================================
        # Step5:
        # 获取发送给LLM的context
        # ==================================

        messages = (
            context_manager
            .get_context_for_llm()
        )


        # ==================================
        # Step6:
        # 验证真实 ContextWindow
        # ==================================

        self.assertEqual(
            len(messages),
            1
        )


        self.assertEqual(
            messages[0]["role"],
            "system"
        )


        self.assertIn(
            "language",
            messages[0]["content"]
        )


        self.assertIn(
            "Python",
            messages[0]["content"]
        )


        # 验证 token 统计生效
        self.assertGreater(
            context_manager.window.total_tokens,
            0
        )


if __name__ == "__main__":
    unittest.main()