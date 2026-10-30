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

# Scalp EEG is tens of microvolts. Anything above 10 mV after conversion to
# volts almost always means the caller passed microvolts but said "V".
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
        Units of ``data``. Values are converted to volts, the SI unit MNE
        uses internally. The conversion is the only change made to values.
    check_amplitude : bool, default=True
        If True, raise when any value exceeds 10 mV after conversion, which
        usually signals a unit mistake. Disable for recordings with large DC
        offsets (e.g. unreferenced BioSemi data).

    Returns
    -------
    raw : mne.io.RawArray
        A new object; ``data`` is copied and never modified.

    Raises
    ------
    TypeError
        If ``sfreq`` is not a real number.
    ValueError
        On wrong dimensionality, transposed-looking input, mismatched or
        duplicate channel names, non-finite values, unknown units, invalid
        sampling frequency, or implausible amplitude.
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

    volts = arr.astype(np.float64, copy=True) * _UNIT_SCALE[units]
    if not np.isfinite(volts).all():
        raise ValueError("data contains NaN or Inf values.")
    if check_amplitude and n_samp and np.abs(volts).max() > _MAX_PLAUSIBLE_ABS_V:
        raise ValueError(
            f"Max |amplitude| is {np.abs(volts).max():.3g} V after converting from "
            f"{units!r}, above the plausible EEG limit of {_MAX_PLAUSIBLE_ABS_V} V. "
            "Check `units` (e.g. units='uV'), or pass check_amplitude=False."
        )

    info = mne.create_info(ch_names=ch_names, sfreq=float(sfreq), ch_types=ch_types)
    return mne.io.RawArray(volts, info, verbose="error")
