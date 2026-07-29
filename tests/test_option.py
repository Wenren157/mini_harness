import os


def test_use_real_llm(set_use_real_llm):

    env_value = os.environ.get(
        "USE_REAL_LLM"
    )
    print(f"fixture use_real_llm = {set_use_real_llm}")

    print(f"env USE_REAL_LLM = {env_value}")

    assert set_use_real_llm == (
        env_value == "1"
    )