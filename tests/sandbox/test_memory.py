import unittest
import tempfile
import os
from mini_harness.infra.memory import LongTermMemory, MemoryManager


# ---------- 模拟 ContextManager 窗口 ----------
class MockWindow:
    def __init__(self, messages=None):
        self.messages = messages if messages is not None else []

    def get_messages_for_llm(self):
        return self.messages


class MockContextManager:
    def __init__(self, messages=None):
        self.window = MockWindow(messages)


# ---------- LongTermMemory 测试 ----------
class TestLongTermMemory(unittest.TestCase):
    def test_lru_eviction(self):
        ltm = LongTermMemory(max_size=3)
        ltm.add("k1", "v1")
        ltm.add("k2", "v2")
        ltm.add("k3", "v3")
        ltm.add("k4", "v4")  # 触发淘汰，淘汰 k1
        self.assertIsNone(ltm.get("k1"))
        self.assertEqual(ltm.get("k2"), "v2")
        self.assertEqual(ltm.get("k3"), "v3")
        self.assertEqual(ltm.get("k4"), "v4")
        self.assertEqual(len(ltm._data), 3)

    def test_get_updates_order(self):
        ltm = LongTermMemory(max_size=3)
        ltm.add("k1", "v1")
        ltm.add("k2", "v2")
        ltm.add("k3", "v3")
        ltm.get("k1")                     # k1 变为最近使用
        ltm.add("k4", "v4")               # 淘汰最久未访问的 k2
        self.assertIsNone(ltm.get("k2"))
        self.assertEqual(ltm.get("k1"), "v1")
        self.assertEqual(ltm.get("k3"), "v3")
        self.assertEqual(ltm.get("k4"), "v4")
        self.assertEqual(len(ltm._data), 3)

    def test_save_and_load(self):
        ltm = LongTermMemory(max_size=5)
        ltm.add("a", "hello")
        ltm.add("b", "world")
        ltm.get("a")  # a 的访问计数变为 1

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "memory.json")
            ltm.save(filepath)

            ltm2 = LongTermMemory(max_size=5)
            ltm2.load(filepath)

            # 加载后先检查计数，确保持久化正确
            data_a = ltm2._data["a"]
            self.assertEqual(data_a.access_count, 1)
            data_b = ltm2._data["b"]
            self.assertEqual(data_b.access_count, 0)

            # 通过 get 验证内容（get 会更新计数，但此处不再作为断言依据）
            self.assertEqual(ltm2.get("a"), "hello")
            self.assertEqual(ltm2.get("b"), "world")

            # （可选）验证计数已相应增加
            self.assertEqual(ltm2._data["a"].access_count, 2)
            self.assertEqual(ltm2._data["b"].access_count, 1)

    def test_load_nonexistent(self):
        ltm = LongTermMemory()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "nonexist.json")
            ltm.load(filepath)  # 文件不存在，应静默忽略
            self.assertEqual(len(ltm._data), 0)


# ---------- MemoryManager 测试 ----------
class TestMemoryManager(unittest.TestCase):
    def test_consolidate(self):
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user msg 1"},
            {"role": "assistant", "content": "assistant msg 1"},
            {"role": "user", "content": "user msg 2"},
            {"role": "assistant", "content": "assistant msg 2"},
            {"role": "user", "content": "user msg 3"},
        ]
        cm = MockContextManager(messages)

        mm = MemoryManager(max_size=10)
        count = mm.consolidate(cm, n=5)
        self.assertEqual(count, 5)

        all_entries = mm.long_term.get_all()
        self.assertEqual(len(all_entries), 1)
        entry = all_entries[0]
        self.assertTrue(entry["key"].startswith("consolidated_"))
        expected_text = " | ".join([
            "user msg 1", "assistant msg 1",
            "user msg 2", "assistant msg 2",
            "user msg 3"
        ])
        self.assertEqual(entry["content"], expected_text)

    def test_get_context_injections(self):
        mm = MemoryManager(max_size=10)
        mm.add("k1", "content1")
        mm.add("k2", "content2")
        injections = mm.get_context_injections()
        self.assertEqual(len(injections), 2)
        self.assertIn("[记忆] k1: content1", injections)
        self.assertIn("[记忆] k2: content2", injections)

    def test_persist_on_add(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "mm.json")
            mm = MemoryManager(persist_path=filepath, max_size=10)
            mm.add("persist_key", "persist_value")
            mm2 = MemoryManager(persist_path=filepath, max_size=10)
            self.assertEqual(mm2.get("persist_key"), "persist_value")


if __name__ == "__main__":
    unittest.main()