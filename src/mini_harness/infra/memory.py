from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import time
import os


# ============================================================
# 1. 长期记忆条目（内部使用）
# ============================================================
@dataclass
class MemoryEntry:
    content: str
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    # TODO: 未来可添加 embedding 向量字段，用于语义检索
    # embedding: Optional[List[float]] = None


# ============================================================
# 2. 长期记忆（LRU 淘汰 + JSON 持久化）
# ============================================================
class LongTermMemory:
    """
    长期记忆：跨会话持久化，采用 LRU 淘汰。
    容量默认 100 条，超出时淘汰最久未访问的条目。
    """

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._data: OrderedDict[str, MemoryEntry] = OrderedDict()
        # TODO: 未来可接入向量数据库（如 sqlite-vec、ChromaDB）替代内存存储
        # self.vector_store = None

    def add(self, key: str, content: str) -> None:
        """添加或更新记忆。若 key 存在则覆盖并移至末尾；若超限则淘汰首位（LRU）。"""
        # 如果 key 已存在，先删除旧条目
        if key in self._data:
            del self._data[key]
        # 写入新条目
        self._data[key] = MemoryEntry(content=content)
        # 如果超限，淘汰最久未访问的条目（OrderedDict 首位）
        if len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def get(self, key: str) -> Optional[str]:
        """获取记忆内容，同时更新访问次数和最后访问时间，并将条目移至末尾。"""
        if key not in self._data:
            return None
        entry = self._data[key]
        entry.access_count += 1
        entry.last_access = time.time()
        self._data.move_to_end(key)
        return entry.content

    def get_all(self) -> List[Dict[str, Any]]:
        """返回所有条目（用于注入上下文或调试），按最近访问排序。"""
        return [
            {"key": k, "content": v.content, "access_count": v.access_count, "last_access": v.last_access}
            for k, v in self._data.items()
        ]

    # TODO: search() - 当前仅支持全量遍历，未来可升级为：
    # - 关键词匹配（基于 content 的简单字符串包含）
    # - 向量检索（需配合 embedding 和向量数据库）
    def search(self, query: str) -> List[Dict[str, Any]]:
        """简单关键词搜索（未来可扩展为向量检索）。"""
        # TODO: 实现关键词过滤：return [entry for entry in self._data.values() if query in entry.content]
        # TODO: 未来升级：调用 self.vector_store.similarity_search(query, top_k=5)
        return []

    def save(self, filepath: str) -> None:
        """持久化到 JSON 文件。"""
        # TODO: 如果未来存储结构升级（含 embedding），需调整序列化逻辑
        data = {
            k: {"content": v.content, "access_count": v.access_count, "last_access": v.last_access}
            for k, v in self._data.items()
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, filepath: str) -> None:
        """从 JSON 文件加载记忆，若文件不存在则忽略。"""
        if not os.path.exists(filepath):
            return
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, raw in data.items():
            self._data[key] = MemoryEntry(
                content=raw["content"],
                access_count=raw.get("access_count", 0),
                last_access=raw.get("last_access", time.time())
            )


# ============================================================
# 3. 记忆管理器（统一入口）
# ============================================================
class MemoryManager:
    """
    统一记忆入口，负责与 ContextManager 协同。
    不自动 consolidation，由调用方（如 Runtime.run() 末尾）手动触发。
    """

    def __init__(self, persist_path: Optional[str] = None, max_size: int = 100):
        self.long_term = LongTermMemory(max_size=max_size)
        self.persist_path = persist_path
        if persist_path and os.path.exists(persist_path):
            self.long_term.load(persist_path)

    def add(self, key: str, content: str) -> None:
        """添加长期记忆（默认持久化）。"""
        self.long_term.add(key, content)
        if self.persist_path:
            self.long_term.save(self.persist_path)

    def get(self, key: str) -> Optional[str]:
        """获取长期记忆。"""
        return self.long_term.get(key)

    def search(self, query: str) -> List[Dict[str, Any]]:
        """搜索记忆（透传 LongTermMemory.search）。"""
        # TODO: 当前为全量遍历，未来可升级为向量检索
        return self.long_term.search(query)

    def consolidate(self, context_manager, n: int = 5) -> int:
        """
        将 ContextManager 中最近 N 条非 system 消息沉淀为长期记忆。
        返回沉淀条数。
        """
        # TODO: 当前仅为简单拼接，未来可接入 LLM 生成摘要（如调用 self.llm.summarize）
        messages = context_manager.window.get_messages_for_llm()
        non_system = [m for m in messages if m.get("role") != "system"]
        recent = non_system[-n:] if len(non_system) >= n else non_system
        if not recent:
            return 0

        summary_text = " | ".join([m.get("content", "") for m in recent])
        key = f"consolidated_{int(time.time())}"
        self.long_term.add(key, summary_text)
        if self.persist_path:
            self.long_term.save(self.persist_path)
        return len(recent)

    def get_context_injections(self) -> List[str]:
        """返回所有长期记忆的文本列表，供 ContextManager 注入到 System Prompt。"""
        return [f"[记忆] {entry['key']}: {entry['content']}" for entry in self.long_term.get_all()]

    def save(self) -> None:
        if self.persist_path:
            self.long_term.save(self.persist_path)

    # TODO: 自动 consolidation 后台任务
    # async def auto_consolidate_loop(self, interval: int = 300):
    #     """后台循环任务，定时将 ContextManager 中的新消息沉淀为长期记忆。"""
    #     pass