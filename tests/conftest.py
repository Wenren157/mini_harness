import pytest
import os

import pytest
import shutil
from pathlib import Path

def pytest_addoption(parser):
    parser.addoption(
        "--use-real-llm",
        action="store_true",
        default=False,
        help="Use real LLM API (requires valid API keys)"
    )

@pytest.fixture(scope="session", autouse=True)
def set_use_real_llm(request):
    """
    获取 pytest 命令行参数 --use-real-llm
    pytest --use-real-llm
        -> True
    pytest
        -> False
    """
    use_real = request.config.getoption("--use-real-llm")
    if use_real:
        os.environ["USE_REAL_LLM"] = "1"
    else:
        os.environ.pop("USE_REAL_LLM", None)
    
    return use_real


@pytest.fixture(autouse=True)
def clean_trace():
    """
    每个测试执行前清理旧 trace 文件，
    防止历史测试产物污染当前测试。
    """
    root = Path(__file__).parent.parent
    trace_dir = root / "traces"
    if trace_dir.exists():
        shutil.rmtree(trace_dir)
    yield