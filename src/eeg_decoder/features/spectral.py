"""Welch power spectral density and frequency-band power, per epoch.

Every value is computed from a single epoch's own samples, so these features
are leakage-free and can be computed before any train/test split. They are
optional and independent of :class:`~eeg_decoder.preprocessing.EEGPipeline`.

Conventions
-----------
* **Explicit resolution.** ``window_s`` is required: the Welch segment length
  sets the frequency resolution ``1 / window_s`` Hz. SciPy's and MNE's
  defaults differ (Hann vs Hamming window, 256-sample segments regardless of
  the sampling rate) and MNE's ``Epochs.compute_psd`` defaults to multitaper,
  so nothing is left to defaults here.
* **Units.** PSD in V^2/Hz; absolute band power in V^2; relative band power
  is a unitless fraction; ``log=True`` gives dB re 1 uV^2.
* **Bands are half-open** ``[low, high)`` and are always supplied by you.
  :data:`CLASSIC_BANDS` is a documented example, never applied implicitly.
* **Band power** is the PSD summed over the bins inside the band times the
  bin width (rectangle rule). Bands that tile a range therefore add up
  exactly to the power of that range.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import mne
import numpy as np
from scipy.signal import welch

from eeg_decoder.features._array import FeatureArray
from eeg_decoder.features._checks import check_epochs, check_number

#: Example band definitions in Hz, half-open [low, high). Pass explicitly if
#: wanted; boundaries vary across the literature (e.g. alpha 8-12 vs 8-13).
CLASSIC_BANDS: Mapping[str, tuple[float, float]] = MappingProxyType({
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
})

_WINDOWS = ("hann", "hamming", "blackman")


def _welch(data: np.ndarray, sfreq: float, window_s: float, overlap: float, window: str):
    window_s = check_number("window_s", window_s)
    overlap = check_number("overlap", overlap, allow_zero=True)
    if overlap >= 1:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}.")
    if window not in _WINDOWS:
        raise ValueError(f"window must be one of {_WINDOWS}, got {window!r}.")
    nperseg = int(round(window_s * sfreq))
    if nperseg < 2:
        raise ValueError(f"window_s={window_s} s gives fewer than 2 samples at {sfreq:g} Hz.")
    if nperseg > data.shape[-1]:
        raise ValueError(
            f"window_s={window_s} s ({nperseg} samples) is longer than the epochs "
            f"({data.shape[-1]} samples, {data.shape[-1] / sfreq:g} s)."
        )
    noverlap = int(round(overlap * nperseg))
    return welch(data, fs=sfreq, window=window, nperseg=nperseg, noverlap=noverlap,
                 detrend="constant", scaling="density", axis=-1)


def welch_psd(
    epochs: mne.BaseEpochs,
    *,
    window_s: float,
    overlap: float = 0.5,
    window: str = "hann",
    fmin: float = 0.0,
    fmax: float | None = None,
    include_bads: bool = False,
) -> FeatureArray:
    """Welch power spectral density of every EEG channel of every epoch.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epoched data. Not modified.
    window_s : float
        Segment length in seconds (required); resolution is ``1 / window_s`` Hz.
        Must not exceed the epoch length.
    overlap : float, default=0.5
        Fraction of overlap between segments, in [0, 1).
    window : {"hann", "hamming", "blackman"}, default="hann"
        Taper applied to each segment.
    fmin, fmax : float
        Frequency range kept (inclusive). ``fmax`` defaults to Nyquist.
    include_bads : bool, default=False
        Include channels marked bad.

    Returns
    -------
    FeatureArray
        dims ``("epoch", "channel", "frequency")``, units ``"V^2/Hz"``.

    Raises
    ------
    TypeError
        If ``epochs`` is not MNE Epochs, or an argument has the wrong type.
    ValueError
        On empty epochs, no EEG channels, NaN/Inf data, a window longer than
        the epochs, or an invalid frequency range.
    """
    data, names, sfreq = check_epochs(epochs, include_bads=include_bads)
    freqs, psd = _welch(data, sfreq, window_s, overlap, window)
    fmin = check_number("fmin", fmin, allow_zero=True)
    fmax = sfreq / 2 if fmax is None else check_number("fmax", fmax)
    if not fmin < fmax <= sfreq / 2:
        raise ValueError(f"Need 0 <= fmin < fmax <= Nyquist ({sfreq / 2:g} Hz), got {fmin}, {fmax}.")
    keep = (freqs >= fmin) & (freqs <= fmax)
    if not keep.any():
        raise ValueError(f"No frequency bins between {fmin} and {fmax} Hz at this resolution.")
    return FeatureArray._from_epochs(
        epochs, psd[..., keep], ("epoch", "channel", "frequency"),
        {"channel": names, "frequency": freqs[keep].tolist()}, "V^2/Hz",
    )


def _check_bands(bands: object, nyquist: float, df: float, allow_overlap: bool) -> dict:
    if not isinstance(bands, Mapping) or not bands:
        raise TypeError("bands must be a non-empty mapping like {'alpha': (8, 13)}.")
    checked = {}
    for name, edges in bands.items():
        if not isinstance(name, str):
            raise TypeError(f"Band names must be strings, got {name!r}.")
        if not isinstance(edges, (tuple, list)) or len(edges) != 2:
            raise TypeError(f"Band {name!r} must be a (low, high) pair, got {edges!r}.")
        low = check_number(f"band {name!r} low", edges[0], allow_zero=True)
        high = check_number(f"band {name!r} high", edges[1])
        if not low < high:
            raise ValueError(f"Band {name!r}: low ({low}) must be below high ({high}).")
        if high > nyquist:
            raise ValueError(f"Band {name!r} reaches {high} Hz, above Nyquist ({nyquist:g} Hz).")
        if high - low < 2 * df:
            raise ValueError(
                f"Band {name!r} ({low}-{high} Hz) is narrower than two frequency bins "
                f"({df:g} Hz each); use a longer window_s."
            )
        checked[name] = (low, high)
    if not allow_overlap:
        spans = sorted((lo, hi, n) for n, (lo, hi) in checked.items())
        for (lo1, hi1, n1), (lo2, hi2, n2) in zip(spans, spans[1:]):
            if lo2 < hi1:
                raise ValueError(
                    f"Bands {n1!r} and {n2!r} overlap; pass allow_overlap=True if intended."
                )
    return checked


def band_power(
    epochs: mne.BaseEpochs,
    bands: Mapping[str, tuple[float, float]],
    *,
    window_s: float,
    relative: bool = False,
    total_range: tuple[float, float] | None = None,
    log: bool = False,
    overlap: float = 0.5,
    window: str = "hann",
    allow_overlap: bool = False,
    include_bads: bool = False,
) -> FeatureArray:
    """Power in each frequency band, per epoch and EEG channel.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epoched data. Not modified.
    bands : mapping of str to (float, float)
        Band name to ``(low, high)`` in Hz, half-open ``[low, high)``. Each
        band must span at least two frequency bins (``2 / window_s`` Hz) and
        stay below Nyquist. See :data:`CLASSIC_BANDS` for an example.
    window_s : float
        Welch segment length in seconds (required); see :func:`welch_psd`.
    relative : bool, default=False
        Divide each band's power by the power in ``total_range`` of the same
        epoch and channel (unitless, leakage-free).
    total_range : (float, float) or None
        Half-open range in Hz for the denominator; required when ``relative``.
    log : bool, default=False
        Return ``10 * log10`` of absolute power in dB re 1 uV^2. Not combined
        with ``relative``.
    overlap, window : see :func:`welch_psd`.
    allow_overlap : bool, default=False
        Allow bands that overlap each other.
    include_bads : bool, default=False
        Include channels marked bad.

    Returns
    -------
    FeatureArray
        dims ``("epoch", "channel", "band")``; units ``"V^2"``, ``"fraction"``,
        or ``"dB re 1 uV^2"``.

    Raises
    ------
    TypeError, ValueError
        On invalid epochs, bands, or option combinations.
    """
    for flag, value in (("relative", relative), ("log", log), ("allow_overlap", allow_overlap)):
        if not isinstance(value, bool):
            raise TypeError(f"{flag} must be True or False.")
    if relative and log:
        raise ValueError("Choose relative=True or log=True, not both.")
    if relative and total_range is None:
        raise ValueError("relative=True needs an explicit total_range, e.g. (1, 45).")
    if not relative and total_range is not None:
        raise ValueError("total_range is only used with relative=True.")

    data, names, sfreq = check_epochs(epochs, include_bads=include_bads)
    freqs, psd = _welch(data, sfreq, window_s, overlap, window)
    df = freqs[1] - freqs[0]
    checked = _check_bands(bands, sfreq / 2, df, allow_overlap)

    def integrate(low: float, high: float) -> np.ndarray:
        inside = (freqs >= low) & (freqs < high)
        return psd[..., inside].sum(axis=-1) * df

    power = np.stack([integrate(lo, hi) for lo, hi in checked.values()], axis=-1)
    units = "V^2"
    if relative:
        total = _check_bands({"total_range": total_range}, sfreq / 2, df, True)["total_range"]
        denom = integrate(*total)
        if np.any(denom <= 0):
            raise ValueError("Total power is zero in some epoch/channel (flat signal).")
        power = power / denom[..., None]
        units = "fraction"
    elif log:
        with np.errstate(divide="ignore"):
            power = 10 * np.log10(power / 1e-12)
        units = "dB re 1 uV^2"
    return FeatureArray._from_epochs(
        epochs, power, ("epoch", "channel", "band"),
        {"channel": names, "band": list(checked)}, units,
    )
