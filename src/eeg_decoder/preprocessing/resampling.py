"""Resampling continuous EEG to a new sampling frequency.

Scientific background
---------------------
* **Aliasing.** After resampling, frequencies above the new Nyquist
  (``sfreq / 2``) would fold back onto lower frequencies (a 70 Hz component
  sampled at 100 Hz would appear at 30 Hz). Both methods remove them first:

  - ``"fft"`` (MNE default) cuts the spectrum at the new Nyquist, an ideal
    brick-wall low-pass. The passband stays flat up to the new Nyquist, but
    a brick-wall filter rings near sharp transients and recording edges.
  - ``"polyphase"`` (``scipy.signal.resample_poly``) uses a FIR
    anti-aliasing filter with a gradual roll-off, so content close to the
    new Nyquist is partly attenuated (measured: a 45 Hz tone keeps ~93 % of
    its amplitude when resampling to 100 Hz).

  Recommended: low-pass first (e.g. ``bandpass(raw, ..., h_freq)`` with
  ``h_freq`` at most about a third of the new rate), so nothing important lies
  near the new Nyquist and the choice of method hardly matters.
  ``info["lowpass"]`` is lowered to the new Nyquist when needed.
* **Event timing.** Stim-channel events are moved to the nearest new sample
  (error at most half a new sample: 5 ms at 100 Hz). MNE resamples stim
  channels specially, so event codes are kept, but events closer together
  than one new sample would merge and one would silently disappear. This
  function raises instead. Annotations are stored in seconds and are
  unaffected.
* **Continuous data only.** Resampling short epochs adds edge artifacts to
  every trial. Downsample epochs by decimation during epoching instead.

Recommended order: filter -> resample -> epoch.
"""

from __future__ import annotations

import math

import mne
import numpy as np

_METHODS = ("fft", "polyphase")


def _stim_events(raw: mne.io.BaseRaw) -> dict[str, np.ndarray]:
    """Event codes per stim channel, as the sequence of transitions to a non-zero value."""
    events = {}
    for idx in mne.pick_types(raw.info, stim=True, exclude=[]):
        x = raw.get_data(picks=[idx])[0]
        prev = np.concatenate(([0.0], x[:-1]))
        events[raw.ch_names[idx]] = x[(x != 0) & (x != prev)]
    return events


def resample(
    raw: mne.io.BaseRaw,
    sfreq: float,
    *,
    method: str = "fft",
) -> mne.io.BaseRaw:
    """Resample a continuous recording to ``sfreq`` Hz, with anti-aliasing.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous recording. Not modified.
    sfreq : float
        New sampling frequency in Hz (finite, > 0). Down- and upsampling are
        both allowed; upsampling adds samples but no information.
    method : {"fft", "polyphase"}, default="fft"
        Resampling algorithm (see module notes).

    Returns
    -------
    raw_resampled : mne.io.BaseRaw
        Resampled copy. ``info["sfreq"]`` and the number of samples change
        (``round(n_samples * sfreq / old_sfreq)``); channel names, types,
        and ``info["bads"]`` are unchanged. Stim channels keep their event
        codes; annotations keep their onsets in seconds.

    Raises
    ------
    TypeError
        If ``raw`` is not continuous MNE data or ``sfreq`` is not a number.
    ValueError
        If ``sfreq`` is not finite and positive, ``method`` is unknown, the
        recording is empty, or stim-channel events would merge.

    Notes
    -----
    To get an events array, extract it *after* resampling (e.g.
    ``mne.find_events(raw_resampled)``), so the sample indices match the new
    sampling rate.
    """
    if isinstance(raw, mne.BaseEpochs):
        raise TypeError(
            "resample applies to continuous data (mne.io.BaseRaw). Resampling "
            "epochs adds edge artifacts to every trial; resample before "
            "epoching, or decimate during epoching."
        )
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(
            f"Expected mne.io.BaseRaw, got {type(raw).__name__}. "
            "Convert arrays with eeg_decoder.io.to_raw first."
        )
    if isinstance(sfreq, bool) or not isinstance(sfreq, (int, float, np.number)):
        raise TypeError(f"sfreq must be a real number, got {type(sfreq).__name__}.")
    if not math.isfinite(sfreq) or sfreq <= 0:
        raise ValueError(f"sfreq must be finite and > 0, got {sfreq}.")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}.")
    if raw.n_times == 0:
        raise ValueError("Recording is empty (0 samples); nothing to resample.")

    before = _stim_events(raw)
    out = raw.copy().load_data()
    out.resample(float(sfreq), method=method, verbose="error")

    for ch, codes in _stim_events(out).items():
        if not np.array_equal(codes, before[ch]):
            raise ValueError(
                f"Resampling to {sfreq} Hz would merge events on stim channel {ch!r} "
                f"({before[ch].size} events before, {codes.size} after): some events "
                f"are closer together than one new sample ({1000 / sfreq:.3g} ms). "
                "Use a higher target sfreq, or keep the events as an array or "
                "annotations taken from the original recording."
            )
    return out
