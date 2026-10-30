"""Threshold-based epoch rejection.

Scientific background
---------------------
Blinks, movements, and electrode pops produce large deflections; a dead or
disconnected electrode produces an almost flat trace. Two simple per-epoch
criteria, both on **peak-to-peak amplitude in volts** per channel type:

* ``reject``: drop the epoch if any channel's peak-to-peak exceeds the limit
  (typical starting points: EEG 100-150 uV, EOG 250 uV);
* ``flat``: drop the epoch if any channel's peak-to-peak is below the limit
  (e.g. EEG 1 uV).

Thresholds are fixed numbers chosen by the analyst, not learned from the
data, so applying them before a train/test split causes no leakage. A
data-driven threshold (e.g. a percentile of all epochs) would have to be
fitted on training folds only; that is out of scope here.

Behaviour verified on MNE 1.13
------------------------------
* Channels in ``info["bads"]`` are ignored by both criteria.
* When every epoch is dropped MNE only warns; here that raises.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping

import mne
import numpy as np

# Same heuristic as eeg_decoder.io.to_raw: thresholds are in volts, so a value
# above 10 mV almost certainly means microvolts were passed (150 instead of 150e-6).
_MAX_PLAUSIBLE_THRESHOLD_V = 1e-2


def _check_thresholds(name: str, thresholds: object, present: set[str]) -> dict | None:
    if thresholds is None:
        return None
    if not isinstance(thresholds, Mapping) or not thresholds:
        raise TypeError(f"{name} must be a non-empty dict like {{'eeg': 150e-6}}, or None.")
    checked = {}
    for ch_type, value in thresholds.items():
        if ch_type not in present:
            raise ValueError(
                f"{name} has a threshold for {ch_type!r}, but the data contain no "
                f"good channels of that type (present: {sorted(present)})."
            )
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise TypeError(f"{name}[{ch_type!r}] must be a number, got {type(value).__name__}.")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name}[{ch_type!r}] must be finite and > 0, got {value}.")
        if value > _MAX_PLAUSIBLE_THRESHOLD_V:
            raise ValueError(
                f"{name}[{ch_type!r}] = {value:g} V is above 10 mV. Thresholds are in "
                f"volts: for 150 uV pass 150e-6, not 150."
            )
        checked[ch_type] = value
    return checked


def reject_epochs(
    epochs: mne.BaseEpochs,
    *,
    reject: Mapping[str, float] | None = None,
    flat: Mapping[str, float] | None = None,
) -> mne.BaseEpochs:
    """Drop epochs whose peak-to-peak amplitude is too large or too small.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epoched data. Not modified.
    reject : dict of {str: float} or None
        Maximum peak-to-peak amplitude in **volts** per channel type,
        e.g. ``{"eeg": 150e-6, "eog": 250e-6}``.
    flat : dict of {str: float} or None
        Minimum peak-to-peak amplitude in **volts** per channel type,
        e.g. ``{"eeg": 1e-6}``. At least one of ``reject``/``flat`` is required.

    Returns
    -------
    epochs_clean : mne.BaseEpochs
        Copy without the rejected epochs. ``epochs_clean.drop_log`` has one
        entry per original event; rejected ones list the offending channels.

    Raises
    ------
    TypeError
        If ``epochs`` is not MNE Epochs or a threshold is malformed.
    ValueError
        If neither criterion is given; a threshold is non-positive, above
        10 mV (likely microvolts passed as volts), or for a channel type with
        no good channels; or if every epoch would be rejected.
    """
    if not isinstance(epochs, mne.BaseEpochs):
        raise TypeError(f"reject_epochs expects mne.Epochs, got {type(epochs).__name__}.")
    if reject is None and flat is None:
        raise ValueError("Give at least one of reject={...} or flat={...}.")
    info = epochs.info
    present = {
        mne.channel_type(info, i)
        for i, name in enumerate(info["ch_names"])
        if name not in info["bads"]
    }
    reject = _check_thresholds("reject", reject, present)
    flat = _check_thresholds("flat", flat, present)

    out = epochs.copy().load_data()
    n_before = len(out)
    if n_before == 0:
        raise ValueError("Epochs object contains no epochs.")
    out.drop_bad(reject=reject, flat=flat, verbose="error")
    if len(out) == 0:
        culprits = Counter(ch for entry in out.drop_log for ch in entry)
        raise ValueError(
            f"All {n_before} epochs would be rejected (reject={reject}, flat={flat}). "
            f"Most frequent channels: {culprits.most_common(5)}. Check the thresholds "
            "(volts) or mark persistently noisy channels as bad first."
        )
    return out
