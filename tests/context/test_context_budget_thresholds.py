from mini_harness.infra.context import ContextWindow


def test_soft_threshold_triggers_compression_before_hard_overflow():
    """
    P1 invariant:

    Soft Compression Threshold 与 Hard Overflow Threshold
    必须是两个不同的边界。

    假设：
        max_tokens = 100
        reserve_tokens = 20
        compression_headroom = 20

    则：
        hard_limit = max_tokens - reserve_tokens = 80
        soft_limit = hard_limit - compression_headroom = 60

    当 total_tokens = 70 时：

        should_compress() == True
        is_overflow() == False

    即：
        已经应该尝试异步 compression，
        但还没有进入需要同步 Hard Trim 的危险区。
    """

    window = ContextWindow(
        max_tokens=100,
        reserve_tokens=20,
        compression_headroom=20,
    )

    # 直接设置 token 数，
    # 避免 TokenEstimator 的估算误差干扰 threshold 单元测试。
    window.total_tokens = 70

    assert window.should_compress() is True
    assert window.is_overflow() is False


def test_below_soft_threshold_does_not_compress_or_overflow():
    """
    P1 invariant:

    Normal Zone:
        total_tokens <= soft_limit

    不应该触发 compression，
    也不应该被视为 hard overflow。
    """

    window = ContextWindow(
        max_tokens=100,
        reserve_tokens=20,
        compression_headroom=20,
    )

    window.total_tokens = 50

    assert window.should_compress() is False
    assert window.is_overflow() is False


def test_above_hard_threshold_is_overflow():
    """
    P1 invariant:

    Emergency Zone:
        total_tokens > hard_limit

    此时既应该满足 soft compression 条件，
    也必须被视为真正的 hard overflow。
    """

    window = ContextWindow(
        max_tokens=100,
        reserve_tokens=20,
        compression_headroom=20,
    )

    window.total_tokens = 90

    assert window.should_compress() is True
    assert window.is_overflow() is True