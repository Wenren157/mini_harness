import unittest

from mini_harness.infra.memory import MemoryManager


# ============================================================
# Mock ContextManager
# ============================================================

class MockContextManager:
    """
    简化版 ContextManager

    真实项目中：
    Memory injection 会进入 system prompt/message window
    """

    def __init__(self):
        self.messages = []

    def inject_memory(self, memories):
        """
        模拟 ContextManager 接收 Memory 注入
        """
        for memory in memories:
            self.messages.append(
                {
                    "role": "system",
                    "content": memory
                }
            )

    def get_messages(self):
        return self.messages



# ============================================================
# Integration Test
# ============================================================

class TestMemoryContextIntegration(unittest.TestCase):

    def test_memory_inject_into_context(self):

        # ============================
        # Step1:
        # 创建长期记忆
        # ============================

        memory_manager = MemoryManager(
            max_size=10
        )

        memory_manager.add(
            "language",
            "Python"
        )


        # ============================
        # Step2:
        # Memory生成Context Injection
        # ============================

        injections = (
            memory_manager
            .get_context_injections()
        )


        self.assertEqual(
            len(injections),
            1
        )


        self.assertIn(
            "language",
            injections[0]
        )

        self.assertIn(
            "Python",
            injections[0]
        )


        # ============================
        # Step3:
        # 注入ContextManager
        # ============================

        context_manager = MockContextManager()

        context_manager.inject_memory(
            injections
        )


        # ============================
        # Step4:
        # 验证Context收到Memory
        # ============================

        messages = (
            context_manager
            .get_messages()
        )


        self.assertEqual(
            len(messages),
            1
        )


        self.assertEqual(
            messages[0]["role"],
            "system"
        )


        self.assertIn(
            "Python",
            messages[0]["content"]
        )


if __name__ == "__main__":
    unittest.main()