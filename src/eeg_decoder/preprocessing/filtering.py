"""Frequency-domain filtering of continuous EEG.

Scientific assumptions (applied identically by both functions)
---------------------------------------------------------------
* **Zero-phase FIR** (MNE defaults: ``firwin`` design, Hamming window,
  automatic filter length and transition bandwidth). Zero phase means no
  time shift of waveforms, so ERP latencies are preserved. The filter is
  non-causal: suitable for offline analysis, not for real-time (online) processing.
* **Continuous data only.** Filtering short epochs puts the filter's edge
  transient inside every trial; filter the recording first, then epoch.
  Passing ``mne.Epochs`` raises ``TypeError``.
* **The recording must be longer than the filter.** MNE only *warns* when
  the FIR filter is longer than the signal and returns distorted output;
  here that case raises ``ValueError``. Required lengths grow as cutoffs
  get sharper: at 250 Hz, a 1-40 Hz band-pass needs 825 samples (3.3 s) and
  a 50 Hz notch needs 1651 samples (6.6 s).
* **No in-place modification.** A filtered copy is returned.
* **Provenance in ``info`` is partial.** MNE updates ``info["highpass"]`` /
  ``info["lowpass"]`` only when *all* data channels are band-pass filtered
  (``picks=None`` or equivalent); with a channel subset they are left
  unchanged. Notch filtering is never recorded in ``info``. A full
  processing record will come from the pipeline configuration (Phase 2d).

Which channels ``picks=None`` filters (verified on MNE 1.13)
------------------------------------------------------------
MNE's "data channels": EEG (and MEG, sEEG, ECoG, DBS, fNIRS), **including
channels listed in** ``info["bads"]``, so bad channels stay comparable and
can later be interpolated. **Not** filtered: EOG, ECG, EMG, stim, and misc
channels. Pass e.g. ``picks=["eeg", "eog"]`` to filter EOG as well. Never
include stim channels in explicit ``picks``: MNE will filter them and
corrupt the event codes.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable, Sequence

import mne
import numpy as np

# Substring of MNE's RuntimeWarning when the FIR filter is longer than the
# data (mne/filter.py). A test pins this so an MNE wording change is caught.
_TOO_SHORT_MSG = "longer than the signal"


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
    if raw.n_times == 0:
        raise ValueError("Recording is empty (0 samples); nothing to filter.")


def _run_mne_filter(apply: Callable[[], object], n_times: int, sfreq: float) -> None:
    """Run an MNE filter call; raise instead of warn if the signal is too short.

    MNE emits its warning only when its log level is WARNING or more verbose,
    so the level is temporarily raised to at least WARNING (never lowered).
    Unrelated warnings are re-emitted unchanged.
    """
    level = min(mne.utils.logger.getEffectiveLevel(), logging.WARNING)
    with mne.use_log_level(level), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        apply()
    for w in caught:
        if _TOO_SHORT_MSG in str(w.message):
            raise ValueError(
                f"Recording too short to filter: {n_times} samples "
                f"({n_times / sfreq:.3g} s). MNE: {w.message} Use a longer "
                "recording, or wider transition bands / higher cutoffs."
            )
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)


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
        Channels to filter, using MNE's convention. ``None`` filters EEG
        channels (MNE "data" channels), *including* channels marked bad, but
        **not** EOG, ECG, EMG, stim, or misc channels (see module notes).
        With a subset of channels, ``info["highpass"/"lowpass"]`` is not updated.

    Returns
    -------
    raw_filtered : mne.io.BaseRaw
        Filtered copy with the same shape, sampling rate, and channels.

    Raises
    ------
    TypeError
        If ``raw`` is not continuous MNE data or a cutoff is not numeric.
    ValueError
        If both cutoffs are None, a cutoff is outside (0, Nyquist),
        ``l_freq >= h_freq``, the recording is empty, or the recording is
        shorter than the FIR filter.

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
    _run_mne_filter(
        lambda: out.filter(
            l_freq, h_freq, picks=picks, method="fir", phase="zero", fir_design="firwin"
        ),
        raw.n_times,
        raw.info["sfreq"],
    )
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
        Channels to filter. ``None`` filters EEG channels (including bad
        ones) but not EOG, ECG, EMG, stim, or misc channels.

    Returns
    -------
    raw_filtered : mne.io.BaseRaw
        Filtered copy.

    Raises
    ------
    TypeError
        If ``raw`` is not continuous MNE data or a frequency is not numeric.
    ValueError
        If ``freqs`` is empty, any frequency is outside (0, Nyquist), the
        recording is empty, or it is shorter than the notch filter.
        Frequencies at or above Nyquist are rejected, not silently dropped.

    Notes
    -----
    **When a notch is redundant.** A line frequency that lies in the
    band-pass *stopband*, i.e. at or above ``h_freq`` plus its transition
    bandwidth (see :func:`bandpass`), is already attenuated by ~50 dB or
    more. Measured at 250 Hz with ``l_freq=1``:

    ======================  ===============  ================
    band-pass ``h_freq``    50 Hz gain       100 Hz gain
    ======================  ===============  ================
    35 Hz (stop >= 43.75)   -60 dB           -79 dB
    40 Hz (stop >= 50)      -50 dB           -88 dB
    45 Hz (stop >= 56.25)   **-4.5 dB**      -73 dB
    ======================  ===============  ================

    So with ``h_freq=40`` an extra 50 Hz notch adds little, but with
    ``h_freq=45`` (or any gamma-band analysis above the line frequency) the
    notch is essential. A notch is also still worth applying when line noise
    is so large that ~50 dB of stopband attenuation leaves visible residue.
    Notch filters are long (1651 samples at 250 Hz for 50 Hz), so they need
    longer recordings and add ringing around sharp transients.
    """
    _require_raw(raw)
    nyquist = raw.info["sfreq"] / 2.0
    freq_list = [freqs] if np.ndim(freqs) == 0 else list(freqs)
    if not freq_list:
        raise ValueError("freqs must contain at least one frequency.")
    checked = np.array([_check_freq("freqs", f, nyquist) for f in freq_list])

    out = raw.copy().load_data()
    _run_mne_filter(
        lambda: out.notch_filter(
            checked, picks=picks, method="fir", phase="zero", fir_design="firwin"
        ),
        raw.n_times,
        raw.info["sfreq"],
    )
    return out
