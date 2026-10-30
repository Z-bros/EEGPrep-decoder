"""Event-locked epoch extraction and baseline correction.

Scientific background
---------------------
An *epoch* is a fixed window ``[tmin, tmax]`` (seconds) around an event, e.g.
``-0.2 .. 0.8`` s around stimulus onset. Epoching should come after
filtering, re-referencing, and resampling of the continuous recording.

Behaviour verified on MNE 1.13, and how this module handles it
----------------------------------------------------------------
* **Windows outside the recording.** MNE drops epochs whose window runs past
  the start or end of the data, *silently* (``drop_log`` says ``NO_DATA`` /
  ``TOO_SHORT``), even with ``on_outside="raise"``. Here this raises,
  listing the affected events, so trials are never lost unnoticed.
* **Decimation aliasing.** MNE only *warns* when ``decim`` gives a sampling
  rate below 3x ``info["lowpass"]``. Here this raises.
* **Baseline.** MNE applies baseline ``(None, 0)`` by default. Here epoching
  never applies a baseline; use :func:`baseline_correct` explicitly.
* **BAD annotations.** Epochs overlapping annotations whose description
  starts with ``"BAD"`` are dropped (MNE default) and recorded in
  ``epochs.drop_log``. These are explicit, user-made marks.
* **Projectors** stored in ``info`` are not applied (``proj=False``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import mne
import numpy as np


def _check_time(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")
    return value


def _check_events(events: object, raw: mne.io.BaseRaw) -> np.ndarray:
    arr = np.asarray(events)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"events must have shape (n_events, 3) [sample, previous, code], got {arr.shape}."
        )
    if arr.shape[0] == 0:
        raise ValueError("events is empty; nothing to epoch.")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"events must be an integer array, got dtype {arr.dtype}.")
    samples = arr[:, 0]
    if np.any(np.diff(samples) < 0):
        raise ValueError("events must be sorted by sample (column 0).")
    if np.any(np.diff(samples) == 0):
        dup = np.unique(samples[1:][np.diff(samples) == 0]).tolist()
        raise ValueError(f"Several events share the same sample: {dup}.")
    outside = samples[(samples < raw.first_samp) | (samples > raw.last_samp)]
    if outside.size:
        raise ValueError(
            f"Event samples {outside.tolist()} lie outside the recording "
            f"[{raw.first_samp}, {raw.last_samp}]. Event samples are absolute "
            "(they include raw.first_samp), as returned by mne.find_events."
        )
    return arr.astype(np.int64, copy=True)


def make_epochs(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    event_id: Mapping[str, int],
    tmin: float,
    tmax: float,
    *,
    decim: int = 1,
) -> mne.Epochs:
    """Cut a continuous recording into epochs around events.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous, preprocessed recording. Not modified.
    events : ndarray of int, shape (n_events, 3)
        MNE events array ``[sample, previous_value, code]`` with absolute
        sample indices (including ``raw.first_samp``), sorted, one event per
        sample. Get it with ``mne.find_events`` or
        ``mne.events_from_annotations`` *after* any resampling.
    event_id : mapping of str to int
        Condition labels to event codes, e.g. ``{"left": 1, "right": 2}``.
        Every code must occur in ``events``; events with other codes are ignored.
    tmin, tmax : float
        Window in seconds relative to each event, ``tmin < tmax``. Both
        endpoints are included, giving ``round((tmax - tmin) * sfreq) + 1``
        samples (before decimation).
    decim : int, default=1
        Keep every ``decim``-th sample. Requires ``sfreq / decim >= 3 *
        info["lowpass"]``; low-pass filter or :func:`resample` first.

    Returns
    -------
    epochs : mne.Epochs
        Preloaded epochs with no baseline applied, all channels kept, and
        ``epochs.events`` / ``epochs.event_id`` / ``epochs.drop_log``
        recording which events were used.

    Raises
    ------
    TypeError
        On wrong input types.
    ValueError
        On empty or malformed events, duplicate event samples, unknown event
        codes, ``tmin >= tmax``, epoch windows outside the recording,
        aliasing ``decim``, or if no epochs remain after BAD annotations.
    """
    if isinstance(raw, mne.BaseEpochs):
        raise TypeError("make_epochs expects continuous data (mne.io.BaseRaw), got Epochs.")
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(
            f"Expected mne.io.BaseRaw, got {type(raw).__name__}. "
            "Convert arrays with eeg_decoder.io.to_raw first."
        )
    if raw.n_times == 0:
        raise ValueError("Recording is empty (0 samples); nothing to epoch.")

    tmin, tmax = _check_time("tmin", tmin), _check_time("tmax", tmax)
    if tmin >= tmax:
        raise ValueError(f"tmin ({tmin}) must be lower than tmax ({tmax}).")

    if isinstance(decim, bool) or not isinstance(decim, (int, np.integer)) or decim < 1:
        raise ValueError(f"decim must be an integer >= 1, got {decim!r}.")
    sfreq = raw.info["sfreq"]
    if decim > 1:
        new_sfreq = sfreq / decim
        lowpass = raw.info["lowpass"]
        if new_sfreq < 3 * lowpass:
            raise ValueError(
                f"decim={decim} gives {new_sfreq:g} Hz, below 3 x info['lowpass'] "
                f"({lowpass:g} Hz): risk of aliasing. Low-pass filter to at most "
                f"{new_sfreq / 3:g} Hz first, or use a smaller decim."
            )

    if not isinstance(event_id, Mapping) or not event_id:
        raise TypeError("event_id must be a non-empty mapping like {'left': 1, 'right': 2}.")
    for name, code in event_id.items():
        if not isinstance(name, str):
            raise TypeError(f"event_id keys must be str, got {name!r}.")
        if isinstance(code, bool) or not isinstance(code, (int, np.integer)):
            raise TypeError(f"event_id[{name!r}] must be an int, got {code!r}.")

    events = _check_events(events, raw)
    present = set(events[:, 2].tolist())
    missing = {k: v for k, v in event_id.items() if v not in present}
    if missing:
        raise ValueError(f"No events found for {missing}; codes present: {sorted(present)}.")

    selected = events[np.isin(events[:, 2], list(event_id.values()))]
    rel = selected[:, 0] - raw.first_samp
    start = rel + int(round(tmin * sfreq))
    stop = rel + int(round(tmax * sfreq))
    out_of_range = selected[(start < 0) | (stop > raw.n_times - 1), 0]
    if out_of_range.size:
        raise ValueError(
            f"Epoch window [{tmin}, {tmax}] s runs outside the recording for events "
            f"at samples {out_of_range.tolist()}. MNE would drop them silently; "
            "remove those events explicitly or shorten the window."
        )

    epochs = mne.Epochs(
        raw, selected, dict(event_id), tmin, tmax,
        baseline=None, preload=True, reject=None, flat=None, proj=False,
        decim=decim, reject_by_annotation=True, on_missing="raise",
        event_repeated="error", verbose="error",
    )
    if len(epochs) == 0:
        raise ValueError(
            "All epochs overlap BAD annotations and were dropped; see the "
            "annotations on the recording."
        )
    return epochs


def baseline_correct(
    epochs: mne.BaseEpochs,
    baseline: tuple[float | None, float | None],
) -> mne.BaseEpochs:
    """Subtract each epoch's mean over a baseline window, per channel.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epoched data. Not modified.
    baseline : tuple of (float or None, float or None)
        Window ``(start, end)`` in seconds within the epoch; ``None`` means
        the epoch's first or last sample, e.g. ``(None, 0)`` for the whole
        pre-stimulus period. Required: there is no default.

    Returns
    -------
    epochs_bc : mne.BaseEpochs
        Copy in which every data channel (EEG, *and* EOG/ECG/EMG) of every
        epoch has zero mean over the window. Stim channels are unchanged.
        ``epochs_bc.baseline`` records the window actually used.

    Raises
    ------
    TypeError
        If ``epochs`` is not MNE Epochs (baseline correction of a continuous
        recording is not meaningful; high-pass filter it instead) or
        ``baseline`` is malformed.
    ValueError
        If the window is reversed, empty, or outside the epoch.

    Notes
    -----
    Each epoch uses only its own samples, so this does not leak information
    across trials. Use a window *before* the event: a baseline that includes
    the response subtracts part of the signal of interest.
    """
    if not isinstance(epochs, mne.BaseEpochs):
        raise TypeError(
            f"baseline_correct expects mne.Epochs, got {type(epochs).__name__}. "
            "Baseline correction of continuous data is not meaningful; use a high-pass filter."
        )
    if not isinstance(baseline, tuple) or len(baseline) != 2:
        raise TypeError(f"baseline must be a (start, end) tuple, got {baseline!r}.")
    b0 = epochs.tmin if baseline[0] is None else _check_time("baseline start", baseline[0])
    b1 = epochs.tmax if baseline[1] is None else _check_time("baseline end", baseline[1])
    if b0 >= b1:
        raise ValueError(f"Baseline start ({b0}) must be lower than its end ({b1}).")
    half = 0.5 / epochs.info["sfreq"]
    if b0 < epochs.tmin - half or b1 > epochs.tmax + half:
        raise ValueError(
            f"Baseline [{b0}, {b1}] s lies outside the epoch [{epochs.tmin}, {epochs.tmax}] s."
        )
    if not np.any((epochs.times >= b0 - half) & (epochs.times <= b1 + half)):
        raise ValueError(f"Baseline [{b0}, {b1}] s contains no samples.")

    out = epochs.copy().load_data()
    if len(out) == 0:
        raise ValueError("Epochs object contains no epochs.")
    out.apply_baseline(baseline, verbose="error")
    return out
