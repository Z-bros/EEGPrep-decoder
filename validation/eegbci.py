"""Real-data validation of EEGPrep-decoder on PhysioNet EEGBCI.

Runs the library end to end on the EEG Motor Movement/Imagery dataset
(Schalk et al. 2004; PhysioNet, ODC-By 1.0) and checks that the results show
known, robust properties of real EEG, not only that the code runs:

==========================  ==================================================
Check                       Pass criterion
==========================  ==================================================
units_in_volts              median EEG channel SD of every raw run in 0.5-500 uV
bandpass_attenuation        a 1-40 Hz band-pass lowers 55-75 Hz power (relative
                            to 8-30 Hz) by >= 30 dB in every subject
notch_line_noise            a 60 Hz notch lowers the 60 Hz line-noise ratio of
                            ``recording_quality`` in every subject
berger_effect               occipital relative alpha (8-13 Hz) is higher with
                            eyes closed (run 2) than open (run 1) in >= 80 %
                            of subjects
motor_erd                   central (C3, Cz, C4) 8-30 Hz power is lower during
                            fist movement (T1/T2, runs 3/7/11) than at rest
                            (T0) in >= 70 % of subjects
ica_reproducible            two ICA fits with the same seed give identical
                            unmixing matrices in every subject
recordings_prepared         every motor run passes ``prepare_recordings``
npz_round_trip              the motor dataset reloads from ``.npz`` exactly
all_subjects_processed      no subject raised an error
==========================  ==================================================

The thresholds are sanity criteria for well-established effects, not
statistical tests; a one-sided sign test p-value is reported alongside.
Bad-channel assessment, ICA component proposals, and epoch rejection are
reported descriptively. Bad channels for the two resting runs are assessed on
the eyes-open run only (see ``_resting_checks``); what the eyes-closed run
alone would have flagged is reported too.

This script is not part of the installed package. It needs ``pooch`` (used by
MNE to download) and network access to physionet.org::

    pip install -e . pooch
    python validation/eegbci.py --subjects 1-10 --out validation-report

Outputs in ``--out``: ``report.json``, ``summary.md``, and figures. The exit
code is 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import mne  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy.signal import welch  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

import eeg_decoder  # noqa: E402
from eeg_decoder.artifacts import (  # noqa: E402
    apply_ica,
    assess_channels,
    find_artifact_components,
    fit_ica,
    mark_bad_channels,
    reject_epochs,
)
from eeg_decoder.dataset import load_npz, prepare_recordings, save_npz  # noqa: E402
from eeg_decoder.epoching import make_epochs  # noqa: E402
from eeg_decoder.features import band_power  # noqa: E402
from eeg_decoder.preprocessing import EEGPipeline, PreprocessingConfig, bandpass, notch  # noqa: E402
from eeg_decoder.quality import plot_psd_comparison, recording_quality  # noqa: E402

DATASET = {
    "name": "EEG Motor Movement/Imagery Dataset (EEGBCI)",
    "url": "https://physionet.org/content/eegmmidb/1.0.0/",
    "license": "Open Data Commons Attribution License v1.0",
    "citation": ("Schalk et al. (2004) IEEE TBME 51(6):1034-1043; "
                 "Goldberger et al. (2000) Circulation 101(23):e215-e220"),
}
EYES_OPEN, EYES_CLOSED, MOTOR = 1, 2, (3, 7, 11)  # runs 3/7/11: open and close left (T1) or right (T2) fist
RUNS = (EYES_OPEN, EYES_CLOSED, *MOTOR)
LINE_FREQ = 60.0  # recorded in the USA
OCCIPITAL, CENTRAL, FRONTAL = ("O1", "Oz", "O2"), ("C3", "Cz", "C4"), ("Fp1", "Fp2")
REJECT = {"eeg": 150e-6}
CLEAN_STEPS = [
    {"op": "notch", "freqs": [LINE_FREQ]},
    {"op": "bandpass", "l_freq": 1.0, "h_freq": 40.0},
    {"op": "interpolate_bad_channels"},
    {"op": "rereference", "ref": "average"},
]
MOTOR_CONFIG = PreprocessingConfig(CLEAN_STEPS + [
    # 0.5-3.5 s after cue onset: skips the onset transient and stays inside the
    # ~4.1 s trial, so no window runs past the end of a run.
    {"op": "make_epochs", "event_source": "annotations",
     "event_id": {"T0": 1, "T1": 2, "T2": 3}, "tmin": 0.5, "tmax": 3.5},
    {"op": "reject_epochs", "reject": REJECT},
])
CRITERIA = {"berger_fraction": 0.8, "erd_fraction": 0.7, "bandpass_db": 30.0}

Loader = Callable[[int, Sequence[int]], Mapping[int, mne.io.BaseRaw]]


# --- data loading -------------------------------------------------------------


def load_eegbci(subject: int, runs: Sequence[int], *, path: str | None = None) -> dict[int, mne.io.BaseRaw]:
    """Download (if needed) and read one subject's runs, with standard channel names."""
    from mne.datasets import eegbci

    if path is not None:
        Path(path).expanduser().mkdir(parents=True, exist_ok=True)
    files = eegbci.load_data(subject, list(runs), path=path, update_path=False)
    out = {}
    for f in files:
        run = int(re.search(r"R(\d+)\.edf$", str(f)).group(1))
        raw = mne.io.read_raw_edf(f, preload=True, verbose="error")
        eegbci.standardize(raw)  # "Fc5." -> "FC5"
        raw.set_montage("colin27_1005")
        out[run] = raw
    return out


# --- small helpers ----------------------------------------------------------------


def _band_ratio_db(raw: mne.io.BaseRaw, num: tuple[float, float], den: tuple[float, float]) -> float:
    """Median over EEG channels of 10*log10(power in ``num`` / power in ``den``)."""
    x = raw.get_data(picks="eeg")
    f, p = welch(x, fs=raw.info["sfreq"], nperseg=int(2 * raw.info["sfreq"]), axis=-1)
    pn = p[:, (f >= num[0]) & (f < num[1])].sum(axis=1)
    pd = p[:, (f >= den[0]) & (f < den[1])].sum(axis=1)
    return float(np.median(10 * np.log10(pn / pd)))


def _flag(raw: mne.io.BaseRaw) -> list[str]:
    """Channels ``assess_channels`` flags, on a 1-40 Hz copy (as recommended)."""
    return list(assess_channels(bandpass(raw, 1.0, 40.0)).bad)


def _clean(raw: mne.io.BaseRaw, bad: Sequence[str]) -> mne.io.BaseRaw:
    clean, _ = EEGPipeline(PreprocessingConfig(CLEAN_STEPS)).run(mark_bad_channels(raw, bad))
    return clean


def _channel_mean(fa, channels: Sequence[str], band: str) -> np.ndarray:
    """Per-epoch mean over ``channels`` of a band_power FeatureArray."""
    names = list(fa.coords["channel"])
    ci = [names.index(c) for c in channels]
    bi = list(fa.coords["band"]).index(band)
    return fa.data[:, ci, bi].mean(axis=1)


def _sign_test(values: Sequence[float], greater: bool) -> dict[str, Any]:
    v = np.asarray(values, float)
    k = int((v > 1).sum() if greater else (v < 1).sum())
    n = int(v.size)
    p = float(binomtest(k, n, 0.5, alternative="greater").pvalue) if n else None
    return {"n_subjects": n, "n_in_expected_direction": k, "fraction": k / n if n else None,
            "sign_test_p": p}


# --- per-subject checks -------------------------------------------------------------


def _resting_checks(
    raws: Mapping[int, mne.io.BaseRaw], *, n_ica: int, seed: int, ica_max_iter: int | str
) -> dict[str, Any]:
    eo, ec = raws[EYES_OPEN], raws[EYES_CLOSED]
    out: dict[str, Any] = {}

    out["median_sd_uv"] = {
        str(r): float(np.median(raw.get_data(picks="eeg").std(axis=1)) * 1e6) for r, raw in raws.items()
    }
    before = _band_ratio_db(eo, (55.0, 75.0), (8.0, 30.0))
    after = _band_ratio_db(bandpass(eo, 1.0, 40.0), (55.0, 75.0), (8.0, 30.0))
    out["bandpass_attenuation_db"] = before - after

    q0 = recording_quality(eo, line_freq=LINE_FREQ).channel_metrics["line_noise_db"]
    q1 = recording_quality(notch(eo, LINE_FREQ), line_freq=LINE_FREQ).channel_metrics["line_noise_db"]
    out["line_noise_db"] = {"before": float(np.median(list(q0.values()))),
                            "after": float(np.median(list(q1.values())))}

    # Bad channels come from the eyes-open run and are applied to both runs.
    # Strong eyes-closed alpha makes occipital channels amplitude outliers, so
    # assessing that run can flag (and interpolation would then erase) the
    # very effect measured here; electrode faults belong to the cap, not the run.
    bad = _flag(eo)
    out["bad_channels"] = {"resting": bad, "flagged_in_eyes_closed_not_applied": _flag(ec)}
    alpha, out["epochs"] = {}, {}
    for name, raw in (("eyes_open", eo), ("eyes_closed", ec)):
        clean = _clean(raw, bad)
        sf = clean.info["sfreq"]
        events = mne.make_fixed_length_events(clean, duration=2.0)
        ep = make_epochs(clean, events, {"rest": 1}, 0.0, 2.0 - 1.0 / sf)
        kept = reject_epochs(ep, reject=REJECT)
        out["epochs"][name] = {"total": len(ep), "kept": len(kept)}
        bp = band_power(kept, {"alpha": (8.0, 13.0)}, window_s=2.0, relative=True,
                        total_range=(1.0, 40.0))
        alpha[name] = float(_channel_mean(bp, OCCIPITAL, "alpha").mean())
        if name == "eyes_open":
            eo_clean = clean
    out["occipital_relative_alpha"] = alpha
    out["alpha_ratio_closed_over_open"] = alpha["eyes_closed"] / alpha["eyes_open"]

    ica = fit_ica(eo_clean, n_components=n_ica, seed=seed, max_iter=ica_max_iter)
    again = fit_ica(eo_clean, n_components=n_ica, seed=seed, max_iter=ica_max_iter)
    out["ica_reproducible"] = bool(np.array_equal(ica.unmixing_matrix_, again.unmixing_matrix_))
    rep = find_artifact_components(ica, eo_clean, eog=list(FRONTAL), measure="correlation",
                                   threshold=0.7)
    ica_out: dict[str, Any] = {"n_components": n_ica, "seed": seed, "proposed": list(rep.eog)}
    if rep.eog:
        removed = apply_ica(eo_clean, ica, exclude=rep.eog)
        ica_out["frontal_1_10hz_change_db"] = (
            _band_ratio_db(removed.copy().pick(list(FRONTAL)), (1.0, 10.0), (20.0, 30.0))
            - _band_ratio_db(eo_clean.copy().pick(list(FRONTAL)), (1.0, 10.0), (20.0, 30.0))
        )
    out["ica"] = ica_out
    return out


def _motor_checks(recordings, out_dir: Path) -> tuple[dict[str, Any], dict[str, float]]:
    ds = prepare_recordings(recordings, MOTOR_CONFIG, on_error="collect", ch_types=["eeg"])
    result: dict[str, Any] = {"failures": [_plain(f) for f in ds.failures], "summary": ds.summary()}

    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        path = save_npz(ds, Path(tmp) / "motor.npz")
        back = load_npz(path)
        result["npz_round_trip"] = bool(
            np.array_equal(back.X, ds.X) and np.array_equal(back.y, ds.y)
            and np.array_equal(back.groups["subject"], ds.groups["subject"])
            and back.records == ds.records
        )

    info = mne.create_info(list(ds.ch_names), ds.sfreq, list(ds.ch_types))
    rest, task = ds.event_id["T0"], (ds.event_id["T1"], ds.event_id["T2"])
    ratios = {}
    for subject in sorted(set(ds.groups["subject"].tolist())):
        mask = ds.groups["subject"] == subject
        ep = mne.EpochsArray(ds.X[mask], info, verbose="error")
        bp = band_power(ep, {"mu_beta": (8.0, 30.0)}, window_s=1.0)
        power = _channel_mean(bp, CENTRAL, "mu_beta")
        y = ds.y[mask]
        ratios[subject] = float(power[np.isin(y, task)].mean() / power[y == rest].mean())
    result["task_over_rest_power"] = ratios
    return result, ratios


# --- report -----------------------------------------------------------------------------


def _plain(obj: Any) -> Any:
    """Read-only mappings and tuples -> plain dicts and lists, for JSON."""
    return json.loads(json.dumps(obj, default=lambda o: dict(o) if isinstance(o, Mapping) else str(o)))


def _check(name: str, criterion: str, passed: bool, details: Any = None) -> dict[str, Any]:
    return {"name": name, "criterion": criterion, "passed": bool(passed), "details": details}


def _effects_figure(alpha: Mapping[str, float], erd: Mapping[str, float]) -> Figure:
    fig = Figure(figsize=(9, 3.6), layout="constrained")
    for ax, values, title, colour in (
        (fig.add_subplot(1, 2, 1), alpha, "Berger effect: occipital alpha, closed / open", "#2a78d6"),
        (fig.add_subplot(1, 2, 2), erd, "Motor ERD: central 8-30 Hz, movement / rest", "#eb6834"),
    ):
        names = list(values)
        ax.bar(range(len(names)), list(values.values()), color=colour)
        ax.axhline(1.0, color="0.3", lw=1, ls="--")
        ax.set_yscale("log")
        for axis in (ax.yaxis.set_major_formatter, ax.yaxis.set_minor_formatter):
            axis(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("ratio (log scale)")
    return fig


def _summary_md(report: Mapping[str, Any]) -> str:
    lines = [f"# EEGPrep-decoder validation on {report['dataset']['name']}", "",
             f"Subjects: {report['subjects']}  ", f"Versions: {report['versions']}", "",
             "| Check | Criterion | Result |", "|---|---|---|"]
    for c in report["checks"]:
        lines.append(f"| {c['name']} | {c['criterion']} | {'pass' if c['passed'] else '**FAIL**'} |")
    if report["errors"]:
        lines += ["", "## Errors", ""] + [f"- subject {s}: {e}" for s, e in report["errors"].items()]
    return "\n".join(lines) + "\n"


def run_validation(
    subjects: Sequence[int],
    out_dir: str | Path,
    *,
    loader: Loader,
    n_ica: int = 20,
    seed: int = 0,
    ica_max_iter: int | str = "auto",
) -> dict[str, Any]:
    """Run every check for ``subjects`` and write the report to ``out_dir``.

    ``loader(subject, runs)`` returns ``{run: raw}`` with standard channel
    names, a montage, data in volts, and ``T0``/``T1``/``T2`` annotations on
    the motor runs; :func:`load_eegbci` does this for the real dataset.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_subject: dict[str, Any] = {}
    errors: dict[str, str] = {}
    motor_recordings = []
    first_psd = None

    for s in subjects:
        sid = f"S{s:03d}"
        try:
            raws = loader(s, RUNS)
            per_subject[sid] = _resting_checks(raws, n_ica=n_ica, seed=seed,
                                                ica_max_iter=ica_max_iter)
            for run in MOTOR:
                bad = _flag(raws[run])
                per_subject[sid]["bad_channels"][f"run_{run}"] = bad
                motor_recordings.append((mark_bad_channels(raws[run], bad), {"subject": sid, "run": run}))
            if first_psd is None:
                eo = raws[EYES_OPEN]
                first_psd = (sid, eo, _clean(eo, per_subject[sid]["bad_channels"]["resting"]))
        except Exception as exc:  # report and continue with the other subjects
            errors[sid] = f"{type(exc).__name__}: {exc}"
            per_subject.pop(sid, None)

    motor, erd = (_motor_checks(motor_recordings, out_dir) if motor_recordings
                  else ({"failures": [], "npz_round_trip": False}, {}))
    alpha = {s: v["alpha_ratio_closed_over_open"] for s, v in per_subject.items()}
    berger, erd_stats = _sign_test(list(alpha.values()), True), _sign_test(list(erd.values()), False)
    subj = per_subject.values()

    checks = [
        _check("units_in_volts", "median EEG channel SD of every run in 0.5-500 uV",
               subj and all(0.5 <= v <= 500 for d in subj for v in d["median_sd_uv"].values())),
        _check("bandpass_attenuation",
               f">= {CRITERIA['bandpass_db']:g} dB less 55-75 Hz power (vs 8-30 Hz) after 1-40 Hz band-pass",
               subj and all(d["bandpass_attenuation_db"] >= CRITERIA["bandpass_db"] for d in subj)),
        _check("notch_line_noise", "60 Hz notch lowers the 60 Hz line-noise ratio",
               subj and all(d["line_noise_db"]["after"] < d["line_noise_db"]["before"] for d in subj)),
        _check("berger_effect",
               f"occipital alpha higher eyes-closed in >= {CRITERIA['berger_fraction']:.0%} of subjects",
               berger["fraction"] is not None and berger["fraction"] >= CRITERIA["berger_fraction"], berger),
        _check("motor_erd",
               f"central 8-30 Hz power lower in movement than rest in "
               f">= {CRITERIA['erd_fraction']:.0%} of subjects",
               erd_stats["fraction"] is not None and erd_stats["fraction"] >= CRITERIA["erd_fraction"],
               erd_stats),
        _check("ica_reproducible", "same seed gives identical ICA unmixing",
               subj and all(d["ica_reproducible"] for d in subj)),
        _check("recordings_prepared", "every motor run passes prepare_recordings",
               bool(motor_recordings) and not motor["failures"], motor["failures"]),
        _check("npz_round_trip", "motor dataset reloads from .npz exactly", motor["npz_round_trip"]),
        _check("all_subjects_processed", "no subject raised an error", not errors, errors),
    ]
    report = {
        "dataset": DATASET,
        "subjects": [f"S{s:03d}" for s in subjects],
        "versions": {"eeg_decoder": eeg_decoder.__version__, "mne": mne.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__,
                     "python": platform.python_version()},
        "parameters": {"runs": list(RUNS), "line_freq": LINE_FREQ, "reject": REJECT,
                       "clean_steps": CLEAN_STEPS, "motor_config": MOTOR_CONFIG.to_dict(),
                       "n_ica": n_ica, "seed": seed,
                       "ica_max_iter": ica_max_iter, "criteria": CRITERIA},
        "checks": checks,
        "passed": all(c["passed"] for c in checks),
        "per_subject": per_subject,
        "motor": motor,
        "errors": errors,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (out_dir / "summary.md").write_text(_summary_md(report))
    if alpha or erd:
        _effects_figure(alpha, erd).savefig(out_dir / "effects.png", dpi=120)
    if first_psd is not None:
        sid, before, after = first_psd
        plot_psd_comparison(before, after, fmax=80.0, labels=("raw", "cleaned")).savefig(
            out_dir / f"psd_{sid}_eyes_open.png", dpi=120)
    return report


def _parse_subjects(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        a, _, b = part.strip().partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    if not out or not all(1 <= s <= 109 for s in out):
        raise argparse.ArgumentTypeError("subjects must be in 1-109, e.g. '1-10' or '1,4,7'.")
    return sorted(set(out))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--subjects", type=_parse_subjects, default=_parse_subjects("1-10"))
    parser.add_argument("--out", type=Path, default=Path("validation-report"))
    parser.add_argument("--data-path", default=None, help="Where EEGBCI is (or will be) stored.")
    parser.add_argument("--n-ica", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    report = run_validation(
        args.subjects, args.out, n_ica=args.n_ica, seed=args.seed,
        loader=lambda s, runs: load_eegbci(s, runs, path=args.data_path),
    )
    print((args.out / "summary.md").read_text())
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
