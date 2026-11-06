"""Runtime and memory benchmarks of EEGPrep-decoder against MNE-Python.

Measures what the library's safety costs: each wrapped operation is timed
next to the equivalent direct MNE call with the same semantics (MNE with
``.copy()``, because the library never modifies its input), and the outputs
are checked for exact agreement. Operations without an MNE equivalent
(bad-channel assessment, artifact annotation, quality metrics, features) are
timed alone. No model accuracy is measured.

Not part of the installed package. Usage::

    pip install -e .
    python benchmarks/run.py --size small --out benchmark-report
    python benchmarks/run.py --size large --out benchmark-report   # ~1 GB per copy; use CI

Sizes (all channels EEG plus one stim channel; synthetic 1/f data, seed 0):

======  ========  ======  ========  ============================
size    channels  rate    duration  float64 data per copy
======  ========  ======  ========  ============================
tiny    8         250 Hz  60 s      ~1 MB (used by the tests)
small   32        250 Hz  5 min     ~19 MB
medium  64        500 Hz  10 min    ~154 MB
large   64        1000 Hz 30 min    ~922 MB
======  ========  ======  ========  ============================

Time is the median of ``--repeats`` runs (wall clock, ``time.perf_counter``).
Peak memory is measured in a separate run with ``tracemalloc``, which sees
NumPy arrays but not scratch buffers inside compiled libraries (e.g. SciPy's
FFT), so it is a lower bound. Writes ``benchmarks.json`` and
``benchmarks.md``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
import scipy

import eeg_decoder
from eeg_decoder.artifacts import annotate_artifacts, assess_channels, reject_epochs
from eeg_decoder.epoching import baseline_correct, make_epochs
from eeg_decoder.features import band_power
from eeg_decoder.preprocessing import (
    EEGPipeline, PreprocessingConfig, bandpass, notch, rereference, resample,
)
from eeg_decoder.quality import recording_quality

SIZES = {  # channels, sampling rate (Hz), duration (s)
    "tiny": (8, 250.0, 60.0),
    "small": (32, 250.0, 300.0),
    "medium": (64, 500.0, 600.0),
    "large": (64, 1000.0, 1800.0),
}
EVENT_EVERY_S = 2.0
TMIN, TMAX = -0.2, 0.8
REJECT = {"eeg": 150e-6}
LINE_FREQ = 50.0


def make_raw(n_channels: int, sfreq: float, seconds: float, seed: int = 0) -> mne.io.RawArray:
    """Synthetic 1/f EEG (volts, ~10 uV) with a stim channel, events every 2 s."""
    rng = np.random.default_rng(seed)
    n = int(round(seconds * sfreq))
    data = np.empty((n_channels + 1, n))
    for i in range(n_channels):  # one channel at a time keeps the peak at one copy
        spectrum = np.fft.rfft(rng.standard_normal(n))
        spectrum[1:] /= np.sqrt(np.arange(1, spectrum.size))
        spectrum[0] = 0
        x = np.fft.irfft(spectrum, n)
        data[i] = x / x.std() * 10e-6 + 3e-6 * np.sin(2 * np.pi * LINE_FREQ * np.arange(n) / sfreq)
    data[-1] = 0
    onsets = np.arange(int(2 * sfreq), n - int(2 * sfreq), int(EVENT_EVERY_S * sfreq))
    data[-1, onsets] = np.resize([1, 2], onsets.size)
    names = [f"EEG{i:03d}" for i in range(n_channels)] + ["STI"]
    info = mne.create_info(names, sfreq, ["eeg"] * n_channels + ["stim"])
    return mne.io.RawArray(data, info, copy=None, verbose="error")  # no extra copy


@dataclass(frozen=True)
class Case:
    name: str
    ours: Callable[[], Any]
    mne: Callable[[], Any] | None = None  # same semantics, called directly
    agree: Callable[[Any, Any], bool] | None = None


def _same_data(a, b) -> bool:
    return np.array_equal(a.get_data(), b.get_data())


def build_cases(raw: mne.io.BaseRaw) -> list[Case]:
    sf = raw.info["sfreq"]
    events = mne.find_events(raw, "STI", shortest_event=1, verbose="error")
    event_id = {"a": 1, "b": 2}
    filtered = bandpass(raw, 1.0, 40.0)
    epochs = make_epochs(filtered, events, event_id, TMIN, TMAX)
    steps = [
        {"op": "notch", "freqs": [LINE_FREQ]},
        {"op": "bandpass", "l_freq": 1.0, "h_freq": 40.0},
        {"op": "rereference", "ref": "average"},
        {"op": "resample", "sfreq": sf / 2},
        {"op": "make_epochs", "event_source": "stim", "stim_channel": "STI",
         "event_id": event_id, "tmin": TMIN, "tmax": TMAX},
        {"op": "baseline_correct", "baseline": [None, 0.0]},
        {"op": "reject_epochs", "reject": REJECT},
    ]
    pipeline = EEGPipeline(PreprocessingConfig(steps))

    def mne_pipeline():
        r = raw.copy().notch_filter([LINE_FREQ], verbose="error").filter(1.0, 40.0, verbose="error")
        r.set_eeg_reference("average", projection=False, verbose="error")
        r.resample(sf / 2, verbose="error")
        ev = mne.find_events(r, "STI", shortest_event=1, verbose="error")
        ep = mne.Epochs(r, ev, event_id, TMIN, TMAX, baseline=(None, 0.0), preload=True,
                        proj=False, verbose="error")
        ep.drop_bad(reject=REJECT, verbose="error")
        return ep

    return [
        Case("bandpass 1-40 Hz", lambda: bandpass(raw, 1.0, 40.0),
             lambda: raw.copy().filter(1.0, 40.0, verbose="error"), _same_data),
        Case(f"notch {LINE_FREQ:g} Hz", lambda: notch(raw, [LINE_FREQ]),
             lambda: raw.copy().notch_filter([LINE_FREQ], verbose="error"), _same_data),
        Case("rereference average", lambda: rereference(raw, "average"),
             lambda: mne.set_eeg_reference(raw.copy(), "average", projection=False,
                                           verbose="error")[0], _same_data),
        Case("resample to half rate", lambda: resample(raw, sf / 2),
             lambda: raw.copy().resample(sf / 2, verbose="error"), _same_data),
        Case("make_epochs", lambda: make_epochs(filtered, events, event_id, TMIN, TMAX),
             lambda: mne.Epochs(filtered, events, event_id, TMIN, TMAX, baseline=None,
                                preload=True, proj=False, verbose="error"), _same_data),
        Case("baseline_correct", lambda: baseline_correct(epochs, (None, 0.0)),
             lambda: epochs.copy().apply_baseline((None, 0.0), verbose="error"), _same_data),
        Case("reject_epochs", lambda: reject_epochs(epochs, reject=REJECT),
             lambda: epochs.copy().drop_bad(reject=REJECT, verbose="error"), _same_data),
        Case("full pipeline (7 steps)", lambda: pipeline.run(raw)[0], mne_pipeline, _same_data),
        Case("assess_channels", lambda: assess_channels(filtered)),
        Case("annotate_artifacts", lambda: annotate_artifacts(filtered, peak=REJECT)),
        Case("recording_quality", lambda: recording_quality(raw, line_freq=LINE_FREQ)),
        Case("band_power (features)",
             lambda: band_power(epochs, {"alpha": (8.0, 13.0), "beta": (13.0, 30.0)}, window_s=1.0)),
    ]


def _time(func: Callable[[], Any], repeats: int) -> float:
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def _peak_mb(func: Callable[[], Any]) -> float:
    gc.collect()
    tracemalloc.start()
    try:
        func()
        return tracemalloc.get_traced_memory()[1] / 2**20
    finally:
        tracemalloc.stop()


def run(size: str, repeats: int = 3, seed: int = 0) -> dict[str, Any]:
    """Run every case at ``size`` and return a JSON-serializable report."""
    n_channels, sfreq, seconds = SIZES[size]
    raw = make_raw(n_channels, sfreq, seconds, seed)
    results = []
    for case in build_cases(raw):
        row: dict[str, Any] = {"case": case.name,
                               "ours_s": _time(case.ours, repeats), "ours_peak_mb": _peak_mb(case.ours)}
        if case.mne is not None:
            row["mne_s"] = _time(case.mne, repeats)
            row["mne_peak_mb"] = _peak_mb(case.mne)
            row["time_ratio"] = row["ours_s"] / row["mne_s"]
            row["identical_output"] = bool(case.agree(case.ours(), case.mne()))
        results.append(row)
        gc.collect()
    return {
        "size": size,
        "data": {"n_channels": n_channels, "sfreq": sfreq, "seconds": seconds,
                 "n_samples": int(raw.n_times), "mb_per_copy": float(raw.get_data().nbytes / 2**20),
                 "seed": seed},
        "repeats": repeats,
        "machine": {"python": platform.python_version(), "platform": platform.platform(),
                    "processor": platform.processor() or platform.machine(),
                    "cpu_count": os.cpu_count()},
        "versions": {"eeg_decoder": eeg_decoder.__version__, "mne": mne.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__},
        "results": results,
    }


def to_markdown(report: dict[str, Any]) -> str:
    d, m = report["data"], report["machine"]
    lines = [
        f"## Benchmarks: {report['size']} ({d['n_channels']} channels, {d['sfreq']:g} Hz, "
        f"{d['seconds'] / 60:g} min; {d['mb_per_copy']:.0f} MB per copy)",
        "",
        f"{m['cpu_count']} CPUs, Python {m['python']}, MNE {report['versions']['mne']}, "
        f"median of {report['repeats']}. Peak memory: tracemalloc, a lower bound.",
        "",
        "| Operation | EEGPrep-decoder | MNE directly | Ratio | Identical output "
        "| Peak MB (ours / MNE) |",
        "|---|---|---|---|---|---|",
    ]
    for r in report["results"]:
        if "mne_s" in r:
            lines.append(f"| {r['case']} | {r['ours_s']:.3f} s | {r['mne_s']:.3f} s | "
                         f"{r['time_ratio']:.2f} | {'yes' if r['identical_output'] else '**no**'} | "
                         f"{r['ours_peak_mb']:.0f} / {r['mne_peak_mb']:.0f} |")
        else:
            lines.append(f"| {r['case']} | {r['ours_s']:.3f} s | - | - | - | "
                         f"{r['ours_peak_mb']:.0f} / - |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--size", choices=list(SIZES), default="small")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("benchmark-report"))
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    with mne.use_log_level("error"):  # MNE's filter-design messages would bury the table
        report = run(args.size, args.repeats, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "benchmarks.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    (args.out / "benchmarks.md").write_text(md)
    print(md)
    return 0 if all(r.get("identical_output", True) for r in report["results"]) else 1


if __name__ == "__main__":
    sys.exit(main())
