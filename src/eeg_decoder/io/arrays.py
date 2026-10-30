"""NumPy array -> ``mne.io.RawArray`` adapter.

This is the single entry point for array data. Every later operation works on
MNE objects, so channel names, channel types, sampling rate, and units are
attached once, here, and then travel with the signal.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import mne
import numpy as np

_UNIT_SCALE = {"V": 1.0, "mV": 1e-3, "uV": 1e-6}

# MNE channel types whose values are voltages. Only these are unit-converted
# and amplitude-checked; stim/misc channels hold codes or arbitrary units.
_VOLTAGE_TYPES = frozenset({"eeg", "eog", "ecg", "emg", "seeg", "ecog", "dbs", "bio"})

# Heuristic, not a physiological limit: referenced scalp EEG is typically
# tens of microvolts and rarely exceeds a few hundred, so 10 mV leaves a
# ~100x margin while still catching the common mistake of passing microvolts
# labelled as volts (a 20 uV signal would appear as 20 V).
_MAX_PLAUSIBLE_ABS_V = 1e-2


def to_raw(
    data: np.ndarray,
    sfreq: float,
    ch_names: Sequence[str],
    ch_types: str | Sequence[str] = "eeg",
    *,
    units: str = "V",
    check_amplitude: bool = True,
) -> mne.io.RawArray:
    """Wrap a continuous EEG array in an ``mne.io.RawArray``.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
        Continuous signal. The axis order is required, not inferred: if the
        array looks transposed relative to ``ch_names`` a ``ValueError`` is
        raised instead of guessing.
    sfreq : float
        Sampling frequency in Hz. Must be finite and positive.
    ch_names : sequence of str
        One unique name per row of ``data``.
    ch_types : str or sequence of str, default="eeg"
        A single MNE channel type for all channels, or one per channel
        (e.g. ``"eeg"``, ``"eog"``, ``"stim"``).
    units : {"V", "mV", "uV"}, default="V"
        Units of the **voltage** channels in ``data`` (EEG, EOG, ECG, EMG,
        sEEG, ECoG, DBS, bio). These are converted to volts, the SI unit MNE
        uses internally; the conversion is the only change made to values.
        Other channel types (stim, misc, ...) are copied unchanged, so event
        codes on a stim channel are never rescaled.
    check_amplitude : bool, default=True
        Sanity check against unit mistakes: raise if any voltage channel
        exceeds 10 mV after conversion. This is a **heuristic, not a
        physiological threshold**. Referenced scalp EEG is typically tens of
        microvolts, so 10 mV mainly catches microvolt data passed with
        ``units="V"``. Legitimate data can exceed it, e.g. recordings with
        large electrode DC offsets (unreferenced BioSemi), intracranial
        channels, or saturated amplifiers. Set ``check_amplitude=False`` for
        such data. The check never alters values; it only accepts or rejects.

    Returns
    -------
    raw : mne.io.RawArray
        A new object; ``data`` is copied and never modified.

    Raises
    ------
    TypeError
        If ``sfreq`` is not a real number.
    ValueError
        On wrong dimensionality, zero channels or zero samples,
        transposed-looking input, mismatched or duplicate channel names,
        non-finite values, unknown units, invalid sampling frequency, or
        (with ``check_amplitude=True``) implausible amplitude.

    Notes
    -----
    Very short recordings (even one sample) are accepted here, because
    whether a recording is long enough depends on the operation applied
    later. The filters in :mod:`eeg_decoder.preprocessing` check that
    themselves.
    """
    if isinstance(sfreq, bool) or not isinstance(sfreq, (int, float, np.number)):
        raise TypeError(f"sfreq must be a real number, got {type(sfreq).__name__}.")
    if not math.isfinite(sfreq) or sfreq <= 0:
        raise ValueError(f"sfreq must be finite and > 0, got {sfreq}.")
    if units not in _UNIT_SCALE:
        raise ValueError(f"units must be one of {tuple(_UNIT_SCALE)}, got {units!r}.")

    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(
            f"data must be 2-D (n_channels, n_samples), got shape {arr.shape}. "
            "Epoched data (n_epochs, n_channels, n_samples) is not continuous."
        )
    if not np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr):
        raise ValueError(f"data must be real-valued numeric, got dtype {arr.dtype}.")

    ch_names = list(ch_names)
    n_ch, n_samp = arr.shape
    if n_ch == 0:
        raise ValueError(f"data has no channels (shape {arr.shape}).")
    if n_samp == 0:
        raise ValueError(f"data has no samples (shape {arr.shape}); recording is empty.")
    if len(ch_names) != n_ch:
        hint = (
            " The array looks transposed; expected (n_channels, n_samples)."
            if len(ch_names) == n_samp
            else ""
        )
        raise ValueError(
            f"len(ch_names)={len(ch_names)} does not match data.shape[0]={n_ch}.{hint}"
        )
    if len(set(ch_names)) != len(ch_names):
        dupes = sorted({c for c in ch_names if ch_names.count(c) > 1})
        raise ValueError(f"Duplicate channel names: {dupes}.")
    if not isinstance(ch_types, str) and len(ch_types) != n_ch:
        raise ValueError(f"len(ch_types)={len(ch_types)} does not match n_channels={n_ch}.")

    types = [ch_types] * n_ch if isinstance(ch_types, str) else list(ch_types)
    is_voltage = np.array([t in _VOLTAGE_TYPES for t in types])

    values = arr.astype(np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("data contains NaN or Inf values.")
    values[is_voltage] *= _UNIT_SCALE[units]
    if check_amplitude and is_voltage.any():
        peak = np.abs(values[is_voltage]).max()
        if peak > _MAX_PLAUSIBLE_ABS_V:
            raise ValueError(
                f"Max |amplitude| of voltage channels is {peak:.3g} V after converting "
                f"from {units!r}, above the 10 mV sanity limit (a heuristic, not a "
                "physiological bound). Check `units` (e.g. units='uV'), or pass "
                "check_amplitude=False for data with large DC offsets."
            )

    info = mne.create_info(ch_names=ch_names, sfreq=float(sfreq), ch_types=ch_types)
    return mne.io.RawArray(values, info, verbose="error")
