"""Tests for the read-only tools used by the architecture-audit demo."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import demo.self_architecture_audit as audit
from mini_harness.infra.context import TokenEstimator

MAX_DEPTH = 2
MAX_ENTRIES = 200
MAX_OUTPUT_TOKENS = 2500


def _registry(workspace: Path, *allowed_prefixes: str):
    return audit.create_readonly_tool_registry(
        str(workspace),
        allowed_prefixes=list(allowed_prefixes),
    )


async def _execute(registry, path: str) -> Dict[str, Any]:
    """Exercise the same public Registry path used by the runtime."""
    return await registry.execute_with_retry("list_directory", path=path)


def _walk(entries: Iterable[Dict[str, Any]]):
    for entry in entries:
        yield entry
        yield from _walk(entry.get("children", []))


def _entry_count(result: Dict[str, Any]) -> int:
    return sum(1 for _ in _walk(result["tree"]))


def _make_symlink(target: Path, link: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"current platform cannot create the required symlink: {exc}")


@pytest.fixture
def audit_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    (workspace / "src" / "package").mkdir(parents=True)
    (workspace / "demo").mkdir()
    (workspace / "private").mkdir()
    (workspace / "src" / "package" / "module.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    (workspace / "demo" / "example.py").write_text(
        "print('demo')\n",
        encoding="utf-8",
    )
    (workspace / "private" / "secret.txt").write_text(
        "secret\n",
        encoding="utf-8",
    )
    return workspace


def test_readonly_tool_schema_requires_explicit_path(audit_workspace: Path):
    registry = _registry(audit_workspace, "src", "demo")
    schemas = {
        item["function"]["name"]: item["function"]
        for item in registry.get_schema()
    }

    assert set(schemas) == {"read_file", "list_directory"}
    assert "write_file" not in schemas
    assert "execute_command" not in schemas
    assert schemas["list_directory"]["parameters"]["required"] == ["path"]


def test_readonly_tool_schema_advertises_allowed_audit_roots(
    audit_workspace: Path,
):
    """T1测试: 两个只读工具都必须向模型暴露准确的允许审计根。"""
    allowed_roots = (
        "src/mini_harness",
        "demo",
    )

    registry = audit.create_readonly_tool_registry(
        str(audit_workspace),
        allowed_prefixes=list(allowed_roots),
    )

    schemas = {
        item["function"]["name"]: item["function"]
        for item in registry.get_schema()
    }

    for tool_name in (
        "read_file",
        "list_directory",
    ):
        schema_text = json.dumps(
            schemas[tool_name],
            ensure_ascii=False,
        )

        for root in allowed_roots:
            assert root in schema_text, (
                f"{tool_name} schema 未暴露允许审计根: {root}"
            )


@pytest.mark.asyncio
async def test_readonly_registry_defaults_to_declared_audit_roots(
    audit_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """T2测试: 默认工具注册表必须使用统一声明的审计根，而不是开放整个 workspace。"""

    assert hasattr(audit, "ALLOWED_AUDIT_ROOTS"), (
        "Demo 尚未声明统一的 ALLOWED_AUDIT_ROOTS"
    )

    monkeypatch.setattr(
        audit,
        "ALLOWED_AUDIT_ROOTS",
        (
            "src",
            "demo",
        ),
    )

    registry = audit.create_readonly_tool_registry(
        str(audit_workspace)
    )

    schemas = {
        item["function"]["name"]: item["function"]
        for item in registry.get_schema()
    }

    schema_text = json.dumps(
        schemas,
        ensure_ascii=False,
    )

    assert "src" in schema_text
    assert "demo" in schema_text
    assert "整个 workspace" not in schema_text

    allowed_result = await registry.execute_with_retry(
        "read_file",
        path="src/package/module.py",
    )
    denied_result = await registry.execute_with_retry(
        "read_file",
        path="private/secret.txt",
    )

    assert allowed_result["error"] is None
    assert denied_result["error"]
    assert "不在允许范围内" in denied_result["error"]


async def _capture_verbose_orchestrator_planner_call():
    """运行无执行步骤的 Orchestrator，并返回传给 Planner 的参数。"""

    class RecordingPlanner:
        def __init__(self):
            self.calls = []

        async def plan(
            self,
            goal,
            available_capabilities=None,
            context=None,
        ):
            self.calls.append(
                {
                    "goal": goal,
                    "available_capabilities": available_capabilities,
                    "context": context,
                }
            )
            return []

    class SummaryLLM:
        async def generate(self, messages, tools=None):
            return {
                "content": "test summary",
                "tool_calls": [],
            }

    class RuntimeStub:
        def __init__(self):
            self.event_bus = []
            self.llm = SummaryLLM()

    planner = RecordingPlanner()
    orchestrator = audit.VerboseOrchestrator(
        runtime=RuntimeStub(),
        planner=planner,
    )
    await orchestrator.run_goal(
        "执行只读架构审计"
    )
    assert len(planner.calls) == 1

    return planner.calls[0]


@pytest.mark.asyncio
async def test_verbose_orchestrator_passes_allowed_roots_to_planner_context():
    """T3测试: Planner 的审计上下文必须明确包含全部允许审计根。"""
    planner_call = await _capture_verbose_orchestrator_planner_call()

    planner_context = planner_call["context"]

    assert isinstance(planner_context, str)

    for root in audit.ALLOWED_AUDIT_ROOTS:
        assert root in planner_context, (
            f"Planner context 未暴露允许审计根: {root}"
        )

@pytest.mark.asyncio
async def test_verbose_orchestrator_advertises_readonly_capabilities():
    """T4测试: Planner 能力描述必须包含只读工具和允许审计根。"""
    planner_call = await _capture_verbose_orchestrator_planner_call()

    available_capabilities = planner_call[
        "available_capabilities"
    ]

    assert isinstance(available_capabilities, list)
    assert available_capabilities, (
        "Planner available_capabilities 仍为空"
    )

    capability_text = json.dumps(
        available_capabilities,
        ensure_ascii=False,
    )

    assert "只读" in capability_text
    assert "read_file" in capability_text
    assert "list_directory" in capability_text

    for root in audit.ALLOWED_AUDIT_ROOTS:
        assert root in capability_text, (
            f"Planner capabilities 未暴露允许审计根: {root}"
        )

@pytest.mark.asyncio
async def test_planner_context_uses_full_workspace_relative_source_paths():
    """T5测试: 核心源码目标必须使用可直接传给工具的完整相对路径。"""
    planner_call = await _capture_verbose_orchestrator_planner_call()

    planner_context = planner_call["context"]

    expected_source_paths = (
        "src/mini_harness/core/runtime.py",
        "src/mini_harness/infra/tools.py",
        "src/mini_harness/infra/context.py",
        "src/mini_harness/mcp/client.py",
        "src/mini_harness/mcp/server.py",
        "src/mini_harness/agents/orchestrator.py",
    )

    for source_path in expected_source_paths:
        assert source_path in planner_context, (
            f"Planner context 缺少完整源码路径: {source_path}"
        )


@pytest.mark.asyncio
async def test_audit_pipeline_reads_real_source_evidence(
    tmp_path: Path,
):
    """T7测试: 完整审计链路必须读取真实源码并将证据回传给 LLM。"""
    workspace = tmp_path / "workspace"
    target_path = "src/mini_harness/core/runtime.py"
    source_marker = "REAL_SOURCE_EVIDENCE_MARKER"

    target_file = workspace / target_path
    target_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    target_file.write_text(
        f"{source_marker} = True\n",
        encoding="utf-8",
    )

    class SingleFilePlanner:
        async def plan(
            self,
            goal,
            available_capabilities=None,
            context=None,
        ):
            assert target_path in context
            assert available_capabilities

            return [
                audit.PlanStep(
                    step_id=1,
                    description=(
                        "使用 read_file 读取并分析 "
                        f"{target_path}"
                    ),
                )
            ]

    class EvidenceLLM:
        def __init__(self):
            self.calls = []

        async def generate(
            self,
            messages,
            tools=None,
        ):
            self.calls.append(
                {
                    "messages": messages,
                    "tools": tools,
                }
            )

            if len(self.calls) == 1:
                return {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_read_runtime",
                            "name": "read_file",
                            "arguments": {
                                "path": target_path,
                            },
                        }
                    ],
                }

            if len(self.calls) == 2:
                return {
                    "content": "已基于真实源码完成分析",
                    "tool_calls": [],
                }

            return {
                "content": "最终审计摘要",
                "tool_calls": [],
            }

    successful_reads = []

    registry = audit.create_readonly_tool_registry(
        workspace=str(workspace),
        success_list=successful_reads,
    )

    llm = EvidenceLLM()

    runtime = audit.HarnessRuntime(
        config=audit.RuntimeConfig(
            max_context_tokens=20000,
            max_iterations=3,
            event_bus_maxlen=100,
            tool_timeout=3.0,
        ),
        llm_client=llm,
        tool_registry=registry,
    )

    orchestrator = audit.VerboseOrchestrator(
        runtime=runtime,
        planner=SingleFilePlanner(),
        success_reads=successful_reads,
    )

    result = await orchestrator.run_goal(
        "读取核心 Runtime 源码并执行只读架构审计"
    )

    assert successful_reads == [
        target_path
    ]

    assert result["results"][1]["final_answer"] == (
        "已基于真实源码完成分析"
    )

    assert len(llm.calls) == 3

    second_request_text = json.dumps(
        llm.calls[1]["messages"],
        ensure_ascii=False,
        default=str,
    )

    assert source_marker in second_request_text

    tool_result_events = [
        event
        for event in runtime.event_bus
        if event.type == audit.EventType.TOOL_CALL_RESULT
    ]

    assert len(tool_result_events) == 1
    assert source_marker in str(
        tool_result_events[0].data
    )


@pytest.mark.asyncio
async def test_list_directory_allows_configured_root_and_descendants(
    audit_workspace: Path,
):
    registry = _registry(audit_workspace, "src")

    root_result = await _execute(registry, "src")
    child_result = await _execute(registry, "src/package")

    assert root_result["error"] is None
    assert child_result["error"] is None
    assert any(entry["name"] == "package" for entry in root_result["tree"])
    assert any(entry["name"] == "module.py" for entry in child_result["tree"])


@pytest.mark.asyncio
async def test_read_file_and_list_directory_share_whitelist_semantics(
    audit_workspace: Path,
):
    registry = _registry(audit_workspace, "src")

    allowed_read = await registry.execute_with_retry(
        "read_file",
        path="src/package/module.py",
    )
    denied_read = await registry.execute_with_retry(
        "read_file",
        path="private/secret.txt",
    )
    denied_list = await _execute(registry, "private")

    assert allowed_read["error"] is None
    assert denied_read["error"]
    assert denied_list["error"]
    assert denied_list["tree"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        ".",
        "private",
        "../workspace/private",
        "src-other",
    ],
)
async def test_list_directory_rejects_root_outside_traversal_and_prefix_collision(
    audit_workspace: Path,
    path: str,
):
    (audit_workspace / "src-other").mkdir(exist_ok=True)
    registry = _registry(audit_workspace, "src")

    result = await _execute(registry, path)

    assert result["error"]
    assert result["tree"] == []


@pytest.mark.asyncio
async def test_policy_rejection_happens_before_directory_walk(
    audit_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    registry = _registry(audit_workspace, "src")
    calls = []

    async def forbidden_walk(*args, **kwargs):
        calls.append((args, kwargs))
        return {"tree": [], "error": None}

    monkeypatch.setattr(audit, "list_directory", forbidden_walk)
    result = await _execute(registry, "private")

    assert result["error"]
    assert result["tree"] == []
    assert calls == []


@pytest.mark.asyncio
async def test_list_directory_rejects_symlink_escape(audit_workspace: Path):
    outside = audit_workspace.parent / "outside"
    outside.mkdir()
    (outside / "outside.txt").write_text("outside\n", encoding="utf-8")
    _make_symlink(outside, audit_workspace / "src" / "escape")
    registry = _registry(audit_workspace, "src")

    result = await _execute(registry, "src/escape")

    assert result["error"]
    assert result["tree"] == []


@pytest.mark.asyncio
async def test_list_directory_returns_stable_directory_first_order(tmp_path: Path):
    workspace = tmp_path / "workspace"
    scan = workspace / "scan"
    scan.mkdir(parents=True)
    for name in ("z-file.txt", "a-file.txt"):
        (scan / name).write_text(name, encoding="utf-8")
    for name in ("z-dir", "a-dir"):
        (scan / name).mkdir()
    registry = _registry(workspace, "scan")

    first = await _execute(registry, "scan")
    second = await _execute(registry, "scan")

    expected = ["a-dir", "z-dir", "a-file.txt", "z-file.txt"]
    assert [entry["name"] for entry in first["tree"]] == expected
    assert first["tree"] == second["tree"]


@pytest.mark.asyncio
async def test_list_directory_reports_depth_truncation_with_drilldown_path(
    tmp_path: Path,
):
    workspace = tmp_path / "workspace"
    deepest = workspace / "scan" / "level1" / "level2" / "level3"
    deepest.mkdir(parents=True)
    (deepest / "deep.txt").write_text("deep", encoding="utf-8")
    registry = _registry(workspace, "scan")

    result = await _execute(registry, "scan")
    entries = list(_walk(result["tree"]))
    level2 = next(entry for entry in entries if entry["name"] == "level2")

    assert result["truncated"] is True
    assert "max_depth" in result["truncation_reasons"]
    assert result["stats"]["max_depth"] == MAX_DEPTH
    assert level2["path"] == "scan/level1/level2"
    assert "children" not in level2
    assert not any(entry["name"] == "level3" for entry in entries)

@pytest.mark.asyncio
async def test_depth_limit_ignores_only_filtered_children(
    tmp_path: Path,
):
    workspace = tmp_path / "workspace"
    level2 = workspace / "scan" / "level1" / "level2"
    level2.mkdir(parents=True)

    (level2 / ".env").write_text(
        "secret",
        encoding="utf-8",
    )

    noise_dir = level2 / "__pycache__"
    noise_dir.mkdir()
    (noise_dir / "ignored.pyc").write_text(
        "noise",
        encoding="utf-8",
    )

    registry = _registry(workspace, "scan")

    result = await _execute(registry, "scan")

    assert "max_depth" not in result["truncation_reasons"]
    assert result["truncated"] is False


@pytest.mark.asyncio
async def test_list_directory_enforces_entry_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace = tmp_path / "workspace"
    scan = workspace / "scan"
    scan.mkdir(parents=True)

    entry_limit = 20
    monkeypatch.setattr(audit, "MAX_ENTRIES", entry_limit)

    for index in range(entry_limit + 5):
        (scan / f"file-{index:03d}.txt").write_text(
            "x",
            encoding="utf-8",
        )

    registry = _registry(workspace, "scan")

    result = await _execute(registry, "scan")

    assert result["truncated"] is True
    assert "max_entries" in result["truncation_reasons"]
    assert result["stats"]["max_entries"] == entry_limit
    assert result["stats"]["returned_entries"] == _entry_count(result)
    assert _entry_count(result) <= entry_limit

@pytest.mark.asyncio
async def test_nested_directory_does_not_exceed_entry_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace = tmp_path / "workspace"
    nested = workspace / "scan" / "nested"
    nested.mkdir(parents=True)

    # 使用较小的entry limit隔离MAX_ENTRIES，
    # 避免MAX_OUTPUT_TOKENS先成为约束条件。
    entry_limit = 20
    monkeypatch.setattr(audit, "MAX_ENTRIES", entry_limit)

    for index in range(entry_limit + 5):
        (nested / f"file-{index:03d}.txt").write_text(
            "x",
            encoding="utf-8",
        )

    registry = _registry(workspace, "scan")

    result = await _execute(registry, "scan")
    actual_entries = _entry_count(result)

    assert result["truncated"] is True
    assert "max_entries" in result["truncation_reasons"]

    # Hard Limit：任何树形结构都不能超过server entry budget。
    assert actual_entries <= entry_limit

    # metadata必须描述真正返回给调用方的tree。
    assert result["stats"]["returned_entries"] == actual_entries

@pytest.mark.asyncio
async def test_list_directory_enforces_budget_on_complete_result(tmp_path: Path):
    workspace = tmp_path / "workspace"
    scan = workspace / "scan"
    scan.mkdir(parents=True)
    for index in range(120):
        name = f"{index:03d}-" + ("long-name-" * 15) + ".txt"
        (scan / name).write_text("x", encoding="utf-8")
    registry = _registry(workspace, "scan")

    result = await _execute(registry, "scan")

    assert result["truncated"] is True
    assert "max_output_tokens" in result["truncation_reasons"]
    assert result["stats"]["max_output_tokens"] == MAX_OUTPUT_TOKENS
    assert TokenEstimator.estimate(str(result)) <= MAX_OUTPUT_TOKENS

@pytest.mark.asyncio
async def test_nested_directory_enforces_complete_result_token_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace = tmp_path / "workspace"
    nested = workspace / "scan" / "nested"
    nested.mkdir(parents=True)

    # 本测试只验证token budget。
    # 提高entry limit，避免MAX_ENTRIES先成为约束条件。
    monkeypatch.setattr(audit, "MAX_ENTRIES", 1000)

    for index in range(300):
        (nested / f"file-{index:03d}.txt").write_text(
            "x",
            encoding="utf-8",
        )

    registry = _registry(workspace, "scan")
    result = await _execute(registry, "scan")
    actual_entries = _entry_count(result)

    assert result["truncated"] is True
    assert "max_output_tokens" in result["truncation_reasons"]
    assert (
        TokenEstimator.estimate(str(result))
        <= MAX_OUTPUT_TOKENS
    )
    assert result["stats"]["returned_entries"] == actual_entries


@pytest.mark.asyncio
async def test_list_directory_does_not_recurse_into_symlink(tmp_path: Path):
    workspace = tmp_path / "workspace"
    real_dir = workspace / "scan" / "real"
    real_dir.mkdir(parents=True)
    (real_dir / "nested.txt").write_text("nested", encoding="utf-8")
    _make_symlink(real_dir, workspace / "scan" / "linked")
    registry = _registry(workspace, "scan")

    result = await _execute(registry, "scan")
    linked = next(entry for entry in result["tree"] if entry["name"] == "linked")

    assert linked["type"] == "symlink"
    assert "children" not in linked


@pytest.mark.asyncio
async def test_list_directory_filters_sensitive_files_and_noise_directories(
    tmp_path: Path,
):
    workspace = tmp_path / "workspace"
    scan = workspace / "scan"
    scan.mkdir(parents=True)
    (scan / "visible.py").write_text("pass\n", encoding="utf-8")
    for name in (".env", "server.pem", "id_rsa"):
        (scan / name).write_text("sensitive", encoding="utf-8")
    for name in (".git", "__pycache__", ".pytest_cache", "node_modules"):
        directory = scan / name
        directory.mkdir()
        (directory / "hidden.txt").write_text("hidden", encoding="utf-8")
    registry = _registry(workspace, "scan")

    result = await _execute(registry, "scan")
    names = {entry["name"] for entry in _walk(result["tree"])}

    assert "visible.py" in names
    assert names.isdisjoint(
        {".env", "server.pem", "id_rsa", ".git", "__pycache__", ".pytest_cache", "node_modules"}
    )

@pytest.mark.asyncio
async def test_list_directory_hides_sensitive_named_symlink(
    tmp_path: Path,
):
    workspace = tmp_path / "workspace"
    scan = workspace / "scan"
    target = workspace / "target.txt"

    scan.mkdir(parents=True)
    target.write_text("target", encoding="utf-8")

    _make_symlink(
        target,
        scan / ".env",
    )

    registry = _registry(workspace, "scan")

    result = await _execute(registry, "scan")
    names = {
        entry["name"]
        for entry in _walk(result["tree"])
    }

    assert ".env" not in names

@pytest.mark.asyncio
async def test_untruncated_result_has_complete_metadata(audit_workspace: Path):
    registry = _registry(audit_workspace, "demo")

    result = await _execute(registry, "demo")

    assert result["error"] is None
    assert result["truncated"] is False
    assert result["truncation_reasons"] == []
    assert result["stats"]["returned_entries"] == _entry_count(result)
    assert result["stats"]["max_depth"] == MAX_DEPTH
    assert result["stats"]["max_entries"] == MAX_ENTRIES
    assert result["stats"]["max_output_tokens"] == MAX_OUTPUT_TOKENS
