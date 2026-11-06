"""Example notebooks are stored without outputs; the docs build executes them.

Stored outputs would go stale silently (and bloat diffs). The `docs` workflow
runs every notebook on each build, so a broken example fails CI.
"""

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "docs" / "examples").glob("*.ipynb"))


def test_there_are_example_notebooks():
    assert len(NOTEBOOKS) >= 2


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_outputs_and_is_listed_in_the_docs(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code, "no code cells"
    assert all(c["outputs"] == [] and c["execution_count"] is None for c in code), (
        f"{path.name} has stored outputs; clear them before committing")
    index = (path.parents[1] / "index.md").read_text(encoding="utf-8")
    assert f"examples/{path.stem}" in index
