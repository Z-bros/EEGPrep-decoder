"""The real-data validation script, run offline on synthetic EEG with known effects.

The real run (PhysioNet EEGBCI) needs a download and happens in the separate
validation workflow; these tests check the script's logic: that it passes when
the effects are present and fails when they are absent.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import mne
import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "validation" / "eegbci.py"
_spec = importlib.util.spec_from_file_location("eegprep_validation_eegbci", _PATH)
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

SF = 160.0
CHANNELS = ["Fp1", "Fp2", "F3", "Fz", "F4", "T7", "C3", "Cz", "C4", "T8",
            "P3", "Pz", "P4", "O1", "Oz", "O2"]
MONTAGE = mne.channels.make_standard_montage("colin27_1005")
POS = np.array([MONTAGE.get_positions()["ch_pos"][c] for c in CHANNELS])


def _weights(centre: str, width: float = 0.05) -> np.ndarray:
    d = np.linalg.norm(POS - POS[CHANNELS.index(centre)], axis=1)
    return np.exp(-(d / width) ** 2)


def _pink(rng, n_samples, n=1):
    spec = np.fft.rfft(rng.normal(size=(n, n_samples)), axis=-1)
    f = np.fft.rfftfreq(n_samples, 1 / SF)
    spec[:, 1:] /= np.sqrt(f[1:])
    spec[:, 0] = 0
    x = np.fft.irfft(spec, n_samples, axis=-1)
    return x / x.std(axis=1, keepdims=True)


def _raw(rng, seconds, *, alpha_uv, mu_gain=None, blinks=False, annotations=None):
    n = int(seconds * SF)
    t = np.arange(n) / SF
    # Spatially smooth background: a few latent sources, smooth weights over the scalp.
    latent = _pink(rng, n, 4) * 10e-6
    mix = np.column_stack([np.ones(len(CHANNELS)), *(POS / np.abs(POS).max(axis=0)).T])
    x = mix @ latent + _pink(rng, n, len(CHANNELS)) * 1.5e-6
    x += np.outer(_weights("Oz"), alpha_uv * 1e-6 * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 6)))
    mu = (np.sin(2 * np.pi * 11.0 * t) + 0.6 * np.sin(2 * np.pi * 20.0 * t)) * 12e-6
    x += np.outer(_weights("Cz", 0.06), mu * (np.ones(n) if mu_gain is None else mu_gain))
    x += 4e-6 * np.sin(2 * np.pi * V.LINE_FREQ * t)  # line noise on every channel
    if blinks:
        for onset in np.arange(2.0, seconds - 1, 4.0):
            x += np.outer(_weights("Fp1", 0.08) + _weights("Fp2", 0.08),
                          120e-6 * np.exp(-((t - onset) / 0.08) ** 2))
    raw = mne.io.RawArray(x, mne.create_info(CHANNELS, SF, "eeg"), verbose="error")
    raw.set_montage(MONTAGE)
    if annotations is not None:
        raw.set_annotations(annotations)
    return raw


def _motor_run(rng, *, task_gain):
    seconds, trial = 125.0, 4.15
    onsets = np.arange(0.0, seconds - trial, trial)
    labels = ["T0" if i % 2 == 0 else ("T1" if i % 4 == 1 else "T2") for i in range(onsets.size)]
    gain = np.ones(int(seconds * SF))
    for o, lab in zip(onsets, labels):
        if lab != "T0":
            gain[int(o * SF):int((o + 4.1) * SF)] = task_gain
    ann = mne.Annotations(onsets, 4.1, labels)
    return _raw(rng, seconds, alpha_uv=5.0, mu_gain=gain, annotations=ann)


def synthetic_loader(*, reversed_effects=False, fail_subject=None):
    """EEG with eyes-closed alpha and movement ERD, or with both reversed."""
    def load(subject, runs):
        if subject == fail_subject:
            raise RuntimeError("download failed")
        rng = np.random.default_rng(subject)
        out = {V.EYES_OPEN: _raw(rng, 40.0, alpha_uv=10.0, blinks=True),
               V.EYES_CLOSED: _raw(rng, 40.0, alpha_uv=2.5 if reversed_effects else 30.0)}
        gain = 2.0 if reversed_effects else 0.25
        out.update({r: _motor_run(rng, task_gain=gain) for r in V.MOTOR})
        assert set(out) == set(runs)
        return out
    return load


FAST = {"n_ica": 8, "ica_max_iter": 30}  # extended Infomax is slow; exactness is unaffected


@pytest.fixture(scope="module")
def passing(tmp_path_factory):
    out = tmp_path_factory.mktemp("report")
    return V.run_validation([1, 2], out, loader=synthetic_loader(), **FAST), out


def test_passes_when_effects_are_present(passing):
    report, _ = passing
    failed = [c["name"] for c in report["checks"] if not c["passed"]]
    assert failed == [] and report["passed"]
    s1 = report["per_subject"]["S001"]
    assert s1["alpha_ratio_closed_over_open"] > 1.5
    assert s1["bandpass_attenuation_db"] >= 30
    assert s1["ica"]["proposed"]  # the synthetic blinks are found
    assert s1["ica"]["frontal_1_10hz_change_db"] < 0  # and removing them lowers frontal power
    assert all(r < 1 for r in report["motor"]["task_over_rest_power"].values())


def test_strong_alpha_would_be_flagged_as_bad_but_is_not_interpolated(passing):
    # The hazard the script avoids: assessing the eyes-closed run flags the
    # occipital channels for their (genuine) alpha amplitude.
    report, _ = passing
    for d in report["per_subject"].values():
        assert set(d["bad_channels"]["flagged_in_eyes_closed_not_applied"]) & {"O1", "Oz", "O2"}
        assert not set(d["bad_channels"]["resting"]) & {"O1", "Oz", "O2"}


def test_report_files_are_complete_and_json_safe(passing):
    report, out = passing
    assert sorted(p.name for p in out.iterdir()) == [
        "effects.png", "psd_S001_eyes_open.png", "report.json", "summary.md"]
    on_disk = json.loads((out / "report.json").read_text())
    assert on_disk["checks"] == json.loads(json.dumps(report["checks"]))
    assert on_disk["versions"]["mne"] == mne.__version__
    assert on_disk["parameters"]["seed"] == 0
    assert "| berger_effect |" in (out / "summary.md").read_text()


def test_fails_when_effects_are_reversed(tmp_path):
    report = V.run_validation([1], tmp_path, loader=synthetic_loader(reversed_effects=True), **FAST)
    failed = {c["name"] for c in report["checks"] if not c["passed"]}
    assert failed == {"berger_effect", "motor_erd"} and not report["passed"]


def test_a_failing_subject_is_reported_and_the_rest_still_run(tmp_path):
    report = V.run_validation([1, 2], tmp_path, loader=synthetic_loader(fail_subject=2), **FAST)
    assert report["errors"] == {"S002": "RuntimeError: download failed"}
    assert list(report["per_subject"]) == ["S001"]
    failed = {c["name"] for c in report["checks"] if not c["passed"]}
    assert failed == {"all_subjects_processed"}


@pytest.mark.parametrize("text, expected", [("1-3", [1, 2, 3]), ("4,1, 2-3", [1, 2, 3, 4]), ("7", [7])])
def test_parse_subjects(text, expected):
    assert V._parse_subjects(text) == expected


@pytest.mark.parametrize("text", ["0-2", "108-110", "a"])
def test_parse_subjects_invalid(text):
    with pytest.raises((argparse.ArgumentTypeError, ValueError)):
        V._parse_subjects(text)
