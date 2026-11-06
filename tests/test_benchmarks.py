"""The benchmark harness, run at its smallest size.

Checks that every wrapped operation gives exactly MNE's output (the
benchmark's agreement column) and that the report is complete; timings
themselves are not asserted, since they depend on the machine.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "run.py"
_spec = importlib.util.spec_from_file_location("eegprep_benchmarks", _PATH)
B = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = B  # dataclasses look their module up here
_spec.loader.exec_module(B)


@pytest.fixture(scope="module")
def report():
    return B.run("tiny", repeats=1)


def test_every_wrapped_operation_matches_mne_exactly(report):
    paired = [r for r in report["results"] if "mne_s" in r]
    assert len(paired) == 8
    assert [r["case"] for r in paired if not r["identical_output"]] == []


def test_report_is_complete_and_json_safe(report):
    assert report["data"]["n_channels"] == 8 and report["data"]["sfreq"] == 250.0
    for r in report["results"]:
        assert r["ours_s"] > 0 and r["ours_peak_mb"] >= 0
    assert json.loads(json.dumps(report)) == report
    md = B.to_markdown(report)
    assert md.count("\n| ") == len(report["results"]) + 1  # header + one row per case


def test_main_writes_files_and_exit_code(tmp_path):
    assert B.main(["--size", "tiny", "--repeats", "1", "--out", str(tmp_path)]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["benchmarks.json", "benchmarks.md"]
    with pytest.raises(SystemExit):
        B.main(["--size", "tiny", "--repeats", "0", "--out", str(tmp_path)])


def test_synthetic_data_is_reproducible_and_in_volts():
    a, b = B.make_raw(4, 250.0, 10.0, seed=3), B.make_raw(4, 250.0, 10.0, seed=3)
    assert (a.get_data() == b.get_data()).all()
    sd = a.get_data(picks="eeg").std(axis=1)
    assert ((sd > 5e-6) & (sd < 20e-6)).all()
