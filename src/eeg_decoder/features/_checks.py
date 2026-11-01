"""Shared input validation for feature functions."""

from __future__ import annotations

import math

import mne
import numpy as np


def check_epochs(epochs: object, *, include_bads: bool) -> tuple[np.ndarray, list[str], float]:
    """Validate epochs and return (data, channel names, sfreq) for EEG channels.

    ``data`` has shape (n_epochs, n_channels, n_times), in volts.
    """
    if isinstance(epochs, mne.io.BaseRaw):
        raise TypeError(
            "Features are computed per epoch: pass mne.Epochs (e.g. from "
            "eeg_decoder.epoching.make_epochs), not continuous data."
        )
    if not isinstance(epochs, mne.BaseEpochs):
        raise TypeError(
            f"Expected mne.Epochs, got {type(epochs).__name__}. For arrays, build "
            "mne.EpochsArray(data, info) with explicit sfreq and channel names first."
        )
    if not isinstance(include_bads, bool):
        raise TypeError("include_bads must be True or False.")
    if len(epochs) == 0:
        raise ValueError("Epochs object contains no epochs.")
    picks = mne.pick_types(epochs.info, eeg=True, exclude=[] if include_bads else "bads")
    if picks.size == 0:
        raise ValueError("No EEG channels to compute features on"
                         + ("" if include_bads else " (bad channels are excluded)") + ".")
    data = epochs.get_data(picks=picks)
    if not np.isfinite(data).all():
        raise ValueError("Epoch data contain NaN or Inf values.")
    names = [epochs.ch_names[i] for i in picks]
    return data, names, float(epochs.info["sfreq"])


def check_number(name: str, value: object, *, positive: bool = True, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")
    if positive and (value < 0 or (value == 0 and not allow_zero)):
        raise ValueError(f"{name} must be {'>= 0' if allow_zero else '> 0'}, got {value}.")
    return value
