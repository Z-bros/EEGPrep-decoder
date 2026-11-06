"""Release consistency: version, changelog, dependency floors, sdist contents."""

import ast
import re
import tomllib
from pathlib import Path

import pytest

import eeg_decoder

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_version_has_a_changelog_entry():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(rf"^## \[{re.escape(eeg_decoder.__version__)}\]", changelog, re.M)


def test_release_workflow_reads_the_version_the_package_reports():
    # release.yml parses __init__.py with ast instead of importing the package.
    # Read the imported package's own file, so this also works on an installed wheel.
    tree = ast.parse(Path(eeg_decoder.__file__).read_text(encoding="utf-8"))
    parsed = next(n.value.value for n in tree.body
                  if isinstance(n, ast.Assign) and n.targets[0].id == "__version__")
    assert parsed == eeg_decoder.__version__


def test_ci_tests_exactly_the_declared_minimum_versions():
    if not (ROOT / ".github").is_dir():
        pytest.skip("CI configuration is not part of the source distribution")
    floors = {}
    for req in PYPROJECT["project"]["dependencies"]:
        name, version = re.fullmatch(r"([A-Za-z0-9_.-]+)>=([0-9.]+)", req).groups()
        floors[name.lower()] = version
    pins = {}
    for line in (ROOT / ".github" / "minimum-versions.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, version = re.fullmatch(r"([A-Za-z0-9_.-]+)==([0-9.]+)\.\*", line.strip()).groups()
            pins[name.lower()] = version
    assert pins == floors
    assert re.search(r'python-version: "3\.11"', (ROOT / ".github" / "workflows" / "tests.yml").read_text())
    assert PYPROJECT["project"]["requires-python"] == ">=3.11"


def test_sdist_includes_what_the_tests_read():
    manifest = (ROOT / "MANIFEST.in").read_text()
    for directory in ("tests", "docs", "validation", "benchmarks"):
        assert f"graft {directory}" in manifest
    assert "include CHANGELOG.md" in manifest


def test_readme_links_are_absolute_for_pypi():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert not re.findall(r"\]\((?!https?://|#)[^)]+\)", readme)
