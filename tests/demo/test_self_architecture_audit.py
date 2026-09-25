"""Tests for the read-only tools used by the architecture-audit demo."""
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
