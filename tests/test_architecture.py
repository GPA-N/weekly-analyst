"""Architecture rules from docs/architecture.md, checked by reading imports with ast (no code is run)."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "weekly_analyst"
MAX_LINES = 300

# package -> the packages it may import from (itself is always allowed). Top-level modules (runner, cli, store) are "app".
ALLOWED = {
    "core": set(),
    "sim": {"core"},
    "data": {"core"},
    "detection": {"core", "data"},
    "analysis": {"core", "data"},
    "report": {"core", "data"},
    "delivery": {"core", "data", "report"},
    "systems": {"core", "data", "detection", "analysis", "report", "delivery"},
}


def _package_of(path: Path) -> str:
    parts = path.relative_to(SRC).parts
    return parts[0] if len(parts) > 1 else "app"


def _imported_packages(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        for name in names:
            parts = name.split(".")
            if parts[0] == "weekly_analyst" and len(parts) > 1:
                found.add(parts[1])
            elif parts[0] == "eval":
                found.add("eval")
    return found


def _sources() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def test_packages_only_import_downward():
    violations = []
    for path in _sources():
        package = _package_of(path)
        if package == "app":
            continue  # runner and cli may import anything
        for imported in _imported_packages(path):
            top_level_module = imported in {"runner", "cli", "store", "config"} or imported not in ALLOWED
            if imported == package or imported in ALLOWED[package]:
                continue
            violations.append(f"{path.relative_to(SRC)} imports weekly_analyst.{imported}" + (" (top level)" if top_level_module else ""))
    assert not violations, "\n".join(violations)


def test_nothing_in_the_package_imports_eval():
    offenders = [str(p.relative_to(SRC)) for p in _sources() if "eval" in _imported_packages(p)]
    assert offenders == []


def test_no_source_file_is_over_300_lines():
    too_long = {str(p.relative_to(SRC)): n for p in _sources() if (n := len(p.read_text(encoding="utf-8").splitlines())) > MAX_LINES}
    assert too_long == {}


def test_every_package_has_a_docstring():
    missing = []
    for init in sorted(SRC.rglob("__init__.py")):
        if "__pycache__" in init.parts:
            continue
        if not ast.get_docstring(ast.parse(init.read_text(encoding="utf-8"))):
            missing.append(str(init.relative_to(SRC)))
    assert missing == []


def test_the_scheduler_order_is_the_spec_order():
    from weekly_analyst.systems import SYSTEMS

    names = [type(system).__name__ for system in SYSTEMS]
    assert names == [
        "ExtractSystem",
        "ValidateSystem",
        "DetectSystem",
        "PlanSystem",
        "InvestigateSystem",
        "DraftSystem",
        "CheckSystem",
        "OutboxSystem",
        "RememberSystem",
    ]
