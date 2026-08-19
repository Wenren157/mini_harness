from pathlib import Path

from mini_harness.agents.scope import AgentScope


def test_agent_scope_workspace_isolation(tmp_path):

    agent_a_workspace = (tmp_path / "agent_a")
    agent_b_workspace = (tmp_path / "agent_b")

    scope_a = AgentScope(
        agent_id="agent_a",
        workspace=str(agent_a_workspace)
    )
    scope_b = AgentScope(
        agent_id="agent_b",
        workspace=str(agent_b_workspace)
    )

    # 两个workspace必须不同
    assert (scope_a.workspace != scope_b.workspace)


    # 写入Agent A空间
    file_a = Path(
        scope_a.workspace
    ) / "data.txt"
    file_a.write_text(
        "hello agent A",
        encoding="utf-8"
    )

    # Agent B不应该看到
    file_b = Path(
        scope_b.workspace
    ) / "data.txt"

    assert file_a.exists()
    assert not file_b.exists()

def test_agent_scope_identity():
    scope = AgentScope(
        agent_id="planner_agent",
        workspace="./workspace"
    )

    assert (scope.agent_id == "planner_agent")

def test_agent_scope_metadata():

    scope = AgentScope(
        agent_id="executor_agent",
        workspace="./workspace",
        metadata={
            "role": "executor"
        }
    )
    assert (scope.metadata["role"]=="executor")