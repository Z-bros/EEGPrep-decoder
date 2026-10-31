"""Marking artifact segments in continuous data as ``BAD_*`` annotations.

Scientific background
---------------------
Short artifacts (blinks, movement, electrode pops) and flat stretches
(amplifier saturation, disconnection) affect a few hundred milliseconds of
an otherwise usable recording. Marking them as annotations whose description
starts with ``"BAD"`` lets :func:`eeg_decoder.epoching.make_epochs` drop only
the epochs that overlap them, and keeps a record of what was excluded.

Detection uses a **sliding-window peak-to-peak amplitude** per channel (in
volts), the same quantity :func:`eeg_decoder.artifacts.reject_epochs` uses
per epoch:

* ``peak``: a sample is flagged when, in the window centred on it, any good
  channel of the given type spans more than the threshold;
* ``flat``: a sample is flagged when, in the window centred on it, any good
  channel of the given type spans less than the threshold, for at least
  ``min_duration_s`` in a row.

Why not ``mne.preprocessing.annotate_amplitude``
-------------------------------------------------
MNE's function compares *consecutive samples* (verified on MNE 1.13): it
catches abrupt steps but misses a smooth 300 uV, ~120 ms blink entirely,
because the blink's largest sample-to-sample change is only ~15 uV at 250 Hz.

Notes
-----
* Detection is per recording and uses no labels, so it causes no leakage.
* Channels in ``info["bads"]`` are ignored; assess and mark bad channels
  first (:func:`assess_channels`), or one broken channel will flag the whole
  recording.
* A flagged span extends half a window either side of the artifact, which
  acts as a natural margin; ``pad_s`` adds more.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import mne
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d

from eeg_decoder.artifacts.rejection import _check_thresholds


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as half-open [start, stop) sample index pairs."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _to_annotations(
    raw: mne.io.BaseRaw, runs: list[tuple[int, int]], description: str
) -> mne.Annotations:
    """Build annotations for data-relative sample runs, in raw's time base.

    With no ``orig_time`` (no measurement date), MNE expects onsets relative
    to the first sample of the data; with one, relative to ``orig_time``, so
    ``raw.first_time`` must be added (verified on MNE 1.13).
    """
    sfreq = raw.info["sfreq"]
    orig_time = raw.annotations.orig_time
    offset = 0.0 if orig_time is None else raw.first_time
    onset = [offset + start / sfreq for start, _ in runs]
    duration = [(stop - start) / sfreq for start, stop in runs]
    return mne.Annotations(onset, duration, [description] * len(runs), orig_time=orig_time)


def annotate_artifacts(
    raw: mne.io.BaseRaw,
    *,
    peak: Mapping[str, float] | None = None,
    flat: Mapping[str, float] | None = None,
    window_s: float = 0.5,
    min_duration_s: float = 0.005,
    pad_s: float = 0.0,
) -> mne.Annotations:
    """Find artifact and flat segments and return them as ``BAD_*`` annotations.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous recording. Not modified.
    peak : dict of {str: float} or None
        Maximum peak-to-peak amplitude in **volts** per channel type within
        any window, e.g. ``{"eeg": 150e-6, "eog": 250e-6}``.
    flat : dict of {str: float} or None
        Minimum peak-to-peak amplitude in **volts** per channel type, e.g.
        ``{"eeg": 1e-6}``. At least one of ``peak``/``flat`` is required.
    window_s : float, default=0.5
        Sliding window length in seconds (at least 2 samples, at most the
        recording length). Longer windows catch slower deflections.
    min_duration_s : float, default=0.005
        Flagged runs shorter than this are ignored.
    pad_s : float, default=0.0
        Extra margin added before and after each span (overlaps are merged).

    Returns
    -------
    annotations : mne.Annotations
        ``BAD_peak`` and ``BAD_flat`` spans in the time base of ``raw``. Not
        attached to ``raw``; use :func:`add_annotations` to do so.

    Raises
    ------
    TypeError
        If ``raw`` is not continuous MNE data, or a threshold is malformed.
    ValueError
        If neither criterion is given, a threshold is invalid (non-positive,
        above 10 mV, or for a channel type with no good channels), the
        window does not fit the recording, or a duration is negative.
    """
    if isinstance(raw, mne.BaseEpochs) or not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(
            f"annotate_artifacts expects continuous data (mne.io.BaseRaw), got {type(raw).__name__}."
        )
    if peak is None and flat is None:
        raise ValueError("Give at least one of peak={...} or flat={...}.")
    if raw.n_times == 0:
        raise ValueError("Recording is empty (0 samples).")
    for name, value in (("window_s", window_s), ("min_duration_s", min_duration_s), ("pad_s", pad_s)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
        if not math.isfinite(value) or value < 0 or (name == "window_s" and value == 0):
            raise ValueError(f"{name} must be finite and {'> 0' if name == 'window_s' else '>= 0'}, got {value}.")

    info = raw.info
    good = [i for i, n in enumerate(info["ch_names"]) if n not in info["bads"]]
    present = {mne.channel_type(info, i) for i in good}
    peak = _check_thresholds("peak", peak, present)
    flat = _check_thresholds("flat", flat, present)

    sfreq = info["sfreq"]
    win = int(round(window_s * sfreq))
    if win < 2 or win > raw.n_times:
        raise ValueError(
            f"window_s={window_s} s gives {win} samples; it must cover at least 2 samples "
            f"and at most the recording ({raw.n_times} samples)."
        )
    min_len = int(math.ceil(min_duration_s * sfreq))
    pad = int(round(pad_s * sfreq))

    def flagged(thresholds: dict[str, float], above: bool) -> np.ndarray:
        mask = np.zeros(raw.n_times, bool)
        for ch_type, limit in thresholds.items():
            idx = [i for i in good if mne.channel_type(info, i) == ch_type]
            x = raw.get_data(picks=idx)
            ptp = maximum_filter1d(x, win, axis=1, mode="nearest") - minimum_filter1d(
                x, win, axis=1, mode="nearest"
            )
            mask |= (ptp > limit).any(axis=0) if above else (ptp < limit).any(axis=0)
        return mask

    def spans(mask: np.ndarray) -> list[tuple[int, int]]:
        runs = [(a, b) for a, b in _runs(mask) if b - a >= max(min_len, 1)]
        if pad:
            runs = [(max(0, a - pad), min(raw.n_times, b + pad)) for a, b in runs]
            merged: list[tuple[int, int]] = []
            for a, b in runs:
                if merged and a <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], b))
                else:
                    merged.append((a, b))
            runs = merged
        return runs

    result = _to_annotations(raw, [], "BAD_peak")
    if peak is not None:
        result = result + _to_annotations(raw, spans(flagged(peak, True)), "BAD_peak")
    if flat is not None:
        result = result + _to_annotations(raw, spans(flagged(flat, False)), "BAD_flat")
    return result


def add_annotations(raw: mne.io.BaseRaw, annotations: mne.Annotations) -> mne.io.BaseRaw:
    """Return a copy of ``raw`` with ``annotations`` added to its existing ones.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous recording. Not modified; existing annotations are kept.
    annotations : mne.Annotations
        New annotations, in the same time base as ``raw`` (e.g. from
        :func:`annotate_artifacts` on this recording).

    Returns
    -------
    raw_annotated : mne.io.BaseRaw
        Copy whose annotations are the union of old and new.

    Raises
    ------
    TypeError
        If the inputs have the wrong types.
    ValueError
        If the time bases (``orig_time``) differ.
    """
    if isinstance(raw, mne.BaseEpochs) or not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(f"add_annotations expects mne.io.BaseRaw, got {type(raw).__name__}.")
    if not isinstance(annotations, mne.Annotations):
        raise TypeError(f"annotations must be mne.Annotations, got {type(annotations).__name__}.")
    if annotations.orig_time != raw.annotations.orig_time:
        raise ValueError(
            f"Annotation time base {annotations.orig_time} differs from the recording's "
            f"{raw.annotations.orig_time}; create them from this recording."
        )
    out = raw.copy()
    out.set_annotations(raw.annotations + annotations)
    return out
