"""Frequency-domain filtering of continuous EEG.

Scientific assumptions (applied identically by both functions)
---------------------------------------------------------------
* **Zero-phase FIR** (MNE defaults: ``firwin`` design, Hamming window,
  automatic filter length and transition bandwidth). Zero phase means no
  time shift of waveforms, so ERP latencies are preserved. The filter is
  non-causal: suitable for offline analysis, not for real-time decoding.
* **Continuous data only.** Filtering short epochs puts the filter's edge
  transient inside every trial; filter the recording first, then epoch.
  Passing ``mne.Epochs`` raises ``TypeError``.
* **No in-place modification.** A filtered copy is returned. MNE records
  band-pass cutoffs in ``info["highpass"]`` / ``info["lowpass"]``; notch
  filtering is not recorded in ``info`` (the pipeline config will be).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import mne
import numpy as np


def _require_raw(raw: object) -> None:
    if isinstance(raw, mne.BaseEpochs):
        raise TypeError(
            "Filtering must be applied to continuous data (mne.io.BaseRaw) before "
            "epoching, to keep filter edge artifacts out of the epochs."
        )
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(
            f"Expected mne.io.BaseRaw, got {type(raw).__name__}. "
            "Convert arrays with eeg_decoder.io.to_raw first."
        )


def _check_freq(name: str, value: float, nyquist: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
    value = float(value)
    if not math.isfinite(value) or not 0 < value < nyquist:
        raise ValueError(f"{name} must satisfy 0 < {name} < Nyquist ({nyquist} Hz), got {value}.")
    return value


def bandpass(
    raw: mne.io.BaseRaw,
    l_freq: float | None,
    h_freq: float | None,
    *,
    picks: str | Sequence[str] | None = None,
) -> mne.io.BaseRaw:
    """Zero-phase FIR band-pass (or one-sided high/low-pass) filter.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous recording. Not modified.
    l_freq : float or None
        High-pass cutoff in Hz (lower passband edge). ``None`` for low-pass only.
    h_freq : float or None
        Low-pass cutoff in Hz (upper passband edge). ``None`` for high-pass only.
        Must be below Nyquist (``sfreq / 2``).
    picks : str, list of str, or None, default=None
        Channels to filter, using MNE's convention; ``None`` means all data
        channels (including channels marked bad, so they stay comparable).

    Returns
    -------
    raw_filtered : mne.io.BaseRaw
        Filtered copy with the same shape, sampling rate, and channels.

    Raises
    ------
    TypeError
        If ``raw`` is not continuous MNE data or a cutoff is not numeric.
    ValueError
        If both cutoffs are None, a cutoff is outside (0, Nyquist), or
        ``l_freq >= h_freq``.

    Notes
    -----
    Cutoffs are passband *edges*, not -3 dB points. Transition bandwidths
    follow MNE's ``"auto"`` rule:

    * high-pass: ``min(max(0.25 * l_freq, 2), l_freq)`` Hz *below* ``l_freq``
      (``l_freq=1`` -> transition 0-1 Hz, -6 dB at 0.5 Hz);
    * low-pass: ``min(max(0.25 * h_freq, 2), nyquist - h_freq)`` Hz *above*
      ``h_freq`` (``h_freq=40`` -> transition 40-50 Hz).

    Energy inside a transition band is only partly attenuated (with
    ``l_freq=1``, a 0.2 Hz drift is reduced by ~22 dB, not removed). Beyond
    it, the Hamming window gives ~53 dB stopband attenuation.
    """
    _require_raw(raw)
    nyquist = raw.info["sfreq"] / 2.0
    if l_freq is None and h_freq is None:
        raise ValueError("At least one of l_freq or h_freq must be given.")
    if l_freq is not None:
        l_freq = _check_freq("l_freq", l_freq, nyquist)
    if h_freq is not None:
        h_freq = _check_freq("h_freq", h_freq, nyquist)
    if l_freq is not None and h_freq is not None and l_freq >= h_freq:
        raise ValueError(f"l_freq ({l_freq}) must be lower than h_freq ({h_freq}).")

    out = raw.copy().load_data()
    out.filter(l_freq, h_freq, picks=picks, method="fir", phase="zero", fir_design="firwin")
    return out


def notch(
    raw: mne.io.BaseRaw,
    freqs: float | Sequence[float],
    *,
    picks: str | Sequence[str] | None = None,
) -> mne.io.BaseRaw:
    """Zero-phase FIR notch filter for line noise.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous recording. Not modified.
    freqs : float or sequence of float
        Frequencies to remove, in Hz. Must be given explicitly (50 Hz in
        most of Asia, Europe, Africa; 60 Hz in the Americas). Harmonics are
        not added automatically; for example use
        ``np.arange(50, raw.info["sfreq"] / 2, 50)``.
    picks : str, list of str, or None, default=None
        Channels to filter; ``None`` means all data channels.

    Returns
    -------
    raw_filtered : mne.io.BaseRaw
        Filtered copy.

    Raises
    ------
    TypeError
        If ``raw`` is not continuous MNE data or a frequency is not numeric.
    ValueError
        If ``freqs`` is empty, or any frequency is outside (0, Nyquist).
        Frequencies at or above Nyquist are rejected, not silently dropped.
    """
    _require_raw(raw)
    nyquist = raw.info["sfreq"] / 2.0
    freq_list = [freqs] if np.ndim(freqs) == 0 else list(freqs)
    if not freq_list:
        raise ValueError("freqs must contain at least one frequency.")
    checked = np.array([_check_freq("freqs", f, nyquist) for f in freq_list])

    out = raw.copy().load_data()
    out.notch_filter(checked, picks=picks, method="fir", phase="zero", fir_design="firwin")
    return out
