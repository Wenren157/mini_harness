import subprocess
import sys
import time

TEST_GROUPS = {

    # 每次修改 Runtime/MCP 必跑
    "smoke": [

        "tests/runtime/test_runtime.py",
        "tests/mcp/test_mcp.py",
        "tests/integration/test_runtime_integration.py",
    ],

    # 模块级回归
    "regression": [

        "tests/runtime",
        "tests/sandbox",

        # Memory
        "tests/integration/test_memory_mock_context.py",
        "tests/integration/test_memory_real_context.py",

        # Day6 Agent Unit
        "tests/agents/test_planner.py",
        "tests/agents/test_executor.py",
        "tests/agents/test_orchestrator.py",
        "tests/agents/test_agent_scope_isolation.py",
        "tests/agents/test_message_bus.py",
    ],

    # 完整链路
    "e2e": [
        "tests/integration/test_e2e_pipeline.py",
        "tests/integration/test_full_pipeline.py",
        "tests/integration/test_runtime_tools.py",

        # Day6 Multi-Agent 集成
        "tests/integration/test_multi_agent_pipeline.py",
        "tests/agents/test_agent_runtime_isolation.py",
        "tests/agents/test_agent_message_flow.py",
    ],

    # 真实模型测试
    "real_llm": [
        "tests/real_llm/test_real_llm_pipeline.py"
    ]
}

def run_group(name, tests):

    print("\n")
    print("=" * 80)
    print(f"RUN TEST GROUP: {name}")
    print("=" * 80)
    start=time.time()
    cmd=[
        sys.executable,
        "-m",
        "pytest",
        "-v",
        "-s",
    ]
    cmd.extend(tests)
    result=subprocess.run(cmd)
    cost=time.time()-start

    if result.returncode !=0:
        print(
            f"""
                FAILED:{name}
                time:{cost:.2f}s
            """
        )
        return False
    print(
        f"""
            PASSED:{name}
            time:{cost:.2f}s
        """
    )
    return True

def main():

    mode="smoke"
    if len(sys.argv)>1:
        mode=sys.argv[1]

    if mode=="all":
        groups=TEST_GROUPS

    elif mode in TEST_GROUPS:
        groups={
            mode:
            TEST_GROUPS[mode]
        }

    else:
        print("Available modes:")
        print(list(TEST_GROUPS.keys()))
        sys.exit(1)

    for name,tests in groups.items():
        if not run_group(name,tests):
            sys.exit(1)

    print("\nALL TESTS PASSED")


if __name__=="__main__":
    main()