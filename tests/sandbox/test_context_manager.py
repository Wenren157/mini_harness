import pytest
from mini_harness.infra.context import ContextManager


@pytest.mark.asyncio
async def test_compression_flag_reset_on_error():

    class FailedLLM:
        async def generate(self, messages):
            raise RuntimeError("mock llm failed")


    manager = ContextManager(
        llm_client=FailedLLM(),
        max_tokens=10
    )

    # 填充足够消息触发压缩
    for i in range(10):
        manager.add_user_message(
            f"message {i} " * 10
        )

    task = manager._background_task

    await task

    assert manager._compression_triggered is False