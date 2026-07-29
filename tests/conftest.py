import pytest
import os

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