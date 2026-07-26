# ============================================================
# 演示：模拟连续消息触发异步后台压缩
# 使用 MockLLMClient 模拟 LLM 调用，验证非阻塞性
# ============================================================

import asyncio
from typing import Dict, List
from mini_harness.infra.context import ContextManager, TokenEstimator


class MockLLMClient:
    """模拟 LLM 客户端，用于演示异步压缩"""
    async def generate(self, messages: List[Dict[str, str]]) -> Dict[str, str]:
        # 模拟异步延迟，展示后台运行
        await asyncio.sleep(0.5)
        # 返回一个简短的摘要
        return {"content": "这是对最近5条消息的摘要。"}


async def demo():
    print("=== 开始演示 ContextManager 异步压缩 ===")
    llm = MockLLMClient()
    # 设置较小的 max_tokens（2000），预留 1000，可用 1000，很快触发压缩
    cm = ContextManager(llm, max_tokens=2000)

    # 添加系统消息（占用少量 token）
    cm.add_system_message("You are a helpful assistant.")

    # 模拟用户连续发送 20 条消息，每条约 50 token（英文），总共约 1000 token，会超限
    for i in range(1, 21):
        msg = f"This is message number {i}. " * 10  # 约 50 token
        cm.add_user_message(msg)
        print(f"[Main] 添加消息 {i:2d}, 当前 token: {cm.window.total_tokens:4d}, "
              f"超限: {cm.window.is_overflow()}, 压缩任务运行中: {not (cm._background_task is None or cm._background_task.done())}")

        # 模拟处理间隔，让异步压缩有机会运行
        await asyncio.sleep(0.1)

    # 等待后台压缩任务完成（如果有）
    if cm._background_task and not cm._background_task.done():
        print("[Main] 等待后台压缩完成...")
        await cm._background_task

    print("\n=== 最终状态 ===")
    print(f"消息总数: {len(cm.window.messages)}")
    print(f"总 token 数: {cm.window.total_tokens}")
    print(f"摘要内容: {cm.summary}")
    print("\n消息列表（角色 + 内容前30字符）:")
    for m in cm.window.messages:
        content_preview = m["content"][:30].replace('\n', ' ')
        print(f"  {m['role']:8s}: {content_preview}...")


if __name__ == "__main__":
    asyncio.run(demo())