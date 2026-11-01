"""Morlet-wavelet time-frequency power, per epoch.

Each epoch is transformed on its own samples (leakage-free). Power is in V^2,
with no baseline normalisation: apply your own (e.g. dB change from a
pre-stimulus window) downstream if wanted.

Choosing parameters
-------------------
* ``freqs`` and ``n_cycles`` are required. A wavelet with ``n`` cycles at
  ``f`` Hz has a temporal width of roughly ``n / f`` s and a spectral width of
  roughly ``f / n`` Hz; ``n_cycles`` proportional to ``freqs`` (e.g.
  ``freqs / 2``) keeps the time window constant.
* MNE's wavelets extend beyond ``n / f`` (e.g. 7 cycles at 10 Hz needs 279
  samples, ~1.1 s, at 250 Hz). MNE raises if any wavelet is longer than the
  epoch; this module lets that error through. Values within half a wavelet
  of either edge of the epoch are contaminated by edge effects: epoch with a
  margin and crop afterwards.
* ``decim`` keeps every ``decim``-th time point *after* the transform, so it
  does not alias the power estimate; it only thins the time axis.
"""

from __future__ import annotations

from collections.abc import Sequence

import mne
import numpy as np
from mne.time_frequency import tfr_array_morlet

from eeg_decoder.features._array import FeatureArray
from eeg_decoder.features._checks import check_epochs


def morlet_tfr(
    epochs: mne.BaseEpochs,
    freqs: Sequence[float],
    *,
    n_cycles: float | Sequence[float],
    decim: int = 1,
    include_bads: bool = False,
) -> FeatureArray:
    """Morlet-wavelet power of every EEG channel in every epoch.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epoched data. Not modified.
    freqs : sequence of float
        Frequencies in Hz, strictly increasing, in (0, Nyquist).
    n_cycles : float or sequence of float
        Cycles per wavelet: one value, or one per frequency (required).
    decim : int, default=1
        Keep every ``decim``-th time point of the output.
    include_bads : bool, default=False
        Include channels marked bad.

    Returns
    -------
    FeatureArray
        dims ``("epoch", "channel", "frequency", "time")``, units ``"V^2"``;
        ``coords["time"]`` holds the epoch times in seconds.

    Raises
    ------
    TypeError, ValueError
        On invalid epochs, frequencies, cycles, or ``decim``; MNE's
        ``ValueError`` if a wavelet is longer than the epochs.
    """
    data, names, sfreq = check_epochs(epochs, include_bads=include_bads)
    f = np.asarray(freqs, dtype=float) if not isinstance(freqs, str) else None
    if f is None or f.ndim != 1 or f.size == 0:
        raise TypeError("freqs must be a non-empty 1-D sequence of numbers.")
    if not np.all(np.isfinite(f)) or f.min() <= 0 or f.max() >= sfreq / 2:
        raise ValueError(f"freqs must be finite and in (0, {sfreq / 2:g}) Hz.")
    if np.any(np.diff(f) <= 0):
        raise ValueError("freqs must be strictly increasing.")
    cycles = np.asarray(n_cycles, dtype=float) if not isinstance(n_cycles, (str, bool)) else None
    if cycles is None or cycles.ndim > 1 or (cycles.ndim == 1 and cycles.size != f.size):
        raise TypeError("n_cycles must be a number or one value per frequency.")
    if not np.all(np.isfinite(cycles)) or np.any(cycles <= 0):
        raise ValueError("n_cycles must be finite and > 0.")
    if isinstance(decim, bool) or not isinstance(decim, (int, np.integer)) or decim < 1:
        raise ValueError(f"decim must be an integer >= 1, got {decim!r}.")

    power = tfr_array_morlet(
        data, sfreq, f, n_cycles=cycles if cycles.ndim else float(cycles),
        output="power", decim=int(decim), verbose="error",
    )
    times = epochs.times[:: int(decim)]
    return FeatureArray._from_epochs(
        epochs, power, ("epoch", "channel", "frequency", "time"),
        {"channel": names, "frequency": f.tolist(), "time": times.tolist()}, "V^2",
    )
