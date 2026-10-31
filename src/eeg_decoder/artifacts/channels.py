"""Bad-channel assessment, marking, and interpolation.

The workflow is deliberately split into three explicit steps, so a human can
review the evidence before any data change:

1. :func:`assess_channels` computes per-channel evidence and returns a
   :class:`ChannelAssessment` (it never modifies the data);
2. :func:`mark_bad_channels` writes the chosen names into ``info["bads"]``;
3. :func:`interpolate_bad_channels` replaces bad EEG channels with spherical-
   spline estimates from their neighbours (needs electrode positions).

Assessment criteria (per recording, EEG channels only, no labels)
-----------------------------------------------------------------
Adapted from the PREP pipeline (Bigdely-Shamlo et al., 2015, Front.
Neuroinform. 9:16), whose default thresholds are used:

* ``nan``: the channel contains NaN or Inf.
* ``flat``: the channel's standard deviation, or its robust standard
  deviation (0.7413 x IQR), is below ``flat_threshold`` volts.
* ``deviation``: the channel's robust amplitude is an outlier across
  channels, ``|robust z| > deviation_threshold``. Catches both unusually
  large (noisy, loose) and unusually small (bridged, dead) channels.
* ``correlation``: in non-overlapping windows, the channel's maximum absolute
  correlation with any other channel falls below ``correlation_threshold``
  in more than ``max_low_correlation_fraction`` of windows. Computed on a
  50 Hz low-passed copy when the data contain content above 50 Hz, as in PREP.
* ``hf_noise``: the ratio of high-frequency (> 50 Hz) to low-frequency
  amplitude is an outlier across channels, ``robust z > hf_noise_threshold``.
  Skipped, with the reason recorded, when the data hold no content above
  50 Hz (``sfreq <= 100`` Hz or ``info["lowpass"] <= 50`` Hz).

Robust z-scores use the median and 0.7413 x IQR across channels, so a few bad
channels cannot hide each other by inflating a mean or standard deviation.
They need enough channels to be meaningful: at least 3 are required and more
than ~16 are recommended.

Run the assessment on data that are high-pass filtered (e.g. >= 1 Hz), so
slow drifts do not dominate amplitude and correlation, and *before*
re-referencing to the average, which would spread a bad channel's signal
into all others. Internal filtering for the criteria acts on copies only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import mne
import numpy as np

_IQR_TO_SD = 0.7413  # IQR of a standard normal = 1.349 = 1 / 0.7413
_HF_SPLIT_HZ = 50.0
CRITERIA = ("nan", "flat", "deviation", "correlation", "hf_noise")


@dataclass(frozen=True)
class ChannelAssessment:
    """Evidence about each EEG channel; does not modify any data.

    Attributes
    ----------
    channels : tuple of str
        EEG channels that were assessed, in recording order.
    bad : tuple of str
        Channels meeting at least one criterion, in recording order.
    reasons : mapping of str to tuple of str
        For each channel in ``bad``, the criteria it met.
    metrics : mapping of str to mapping of str to float
        Per criterion, the per-channel value that was thresholded (e.g. the
        robust z-score for ``deviation``). NaN where not computable.
    skipped : mapping of str to str
        Criteria that were not evaluated, with the reason.
    previously_bad : tuple of str
        EEG channels already in ``info["bads"]`` when assessed.
    params : mapping of str to float
        Thresholds and window length used.
    """

    channels: tuple[str, ...]
    bad: tuple[str, ...]
    reasons: Mapping[str, tuple[str, ...]]
    metrics: Mapping[str, Mapping[str, float]]
    skipped: Mapping[str, str]
    previously_bad: tuple[str, ...]
    params: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable copy (NaN metrics become ``None``)."""

        def clean(v: float) -> float | None:
            return None if not math.isfinite(v) else float(v)

        return {
            "channels": list(self.channels),
            "bad": list(self.bad),
            "reasons": {ch: list(r) for ch, r in self.reasons.items()},
            "metrics": {c: {ch: clean(v) for ch, v in m.items()} for c, m in self.metrics.items()},
            "skipped": dict(self.skipped),
            "previously_bad": list(self.previously_bad),
            "params": dict(self.params),
        }

    def summary(self) -> str:
        """Human-readable, one line per flagged channel."""
        if not self.bad:
            return f"No bad channels among {len(self.channels)} assessed."
        lines = [f"{len(self.bad)} of {len(self.channels)} channels flagged:"]
        lines += [f"  {ch}: {', '.join(self.reasons[ch])}" for ch in self.bad]
        lines += [f"  (skipped {c}: {why})" for c, why in self.skipped.items()]
        return "\n".join(lines)


def _robust_z(values: np.ndarray) -> np.ndarray:
    """(x - median) / (0.7413 * IQR); 0 where the spread is zero and x is the median."""
    med = np.median(values)
    q75, q25 = np.percentile(values, [75, 25])
    scale = _IQR_TO_SD * (q75 - q25)
    if scale == 0:
        return np.where(values == med, 0.0, np.inf * np.sign(values - med))
    return (values - med) / scale


def _check_positive(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value}.")
    return value


def assess_channels(
    raw: mne.io.BaseRaw,
    *,
    flat_threshold: float = 1e-15,
    deviation_threshold: float = 5.0,
    correlation_threshold: float = 0.4,
    correlation_window_s: float = 1.0,
    max_low_correlation_fraction: float = 0.01,
    hf_noise_threshold: float = 5.0,
) -> ChannelAssessment:
    """Assess every EEG channel of a continuous recording for bad-channel evidence.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous recording, ideally high-pass filtered and not yet average
        referenced. Not modified.
    flat_threshold : float, default=1e-15
        Volts. Standard deviation (or robust SD) below this means flat.
    deviation_threshold : float, default=5.0
        Absolute robust z-score of channel amplitude above which a channel deviates.
    correlation_threshold : float, default=0.4
        Minimum acceptable maximum |correlation| with any other channel per window.
    correlation_window_s : float, default=1.0
        Window length in seconds for the correlation criterion.
    max_low_correlation_fraction : float, default=0.01
        A channel is bad if more than this fraction of windows is poorly correlated.
    hf_noise_threshold : float, default=5.0
        Robust z-score of the high/low-frequency amplitude ratio.

    Returns
    -------
    ChannelAssessment
        Evidence and the flagged channels. Use :func:`mark_bad_channels` to act on it.

    Raises
    ------
    TypeError
        If ``raw`` is not continuous MNE data, or a threshold is not a number.
    ValueError
        If fewer than 3 EEG channels are present, the recording is empty, or a
        threshold is not finite and positive (fractions must also be < 1).
    """
    if isinstance(raw, mne.BaseEpochs) or not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(
            f"assess_channels expects continuous data (mne.io.BaseRaw), got {type(raw).__name__}."
        )
    params = {
        "flat_threshold": _check_positive("flat_threshold", flat_threshold),
        "deviation_threshold": _check_positive("deviation_threshold", deviation_threshold),
        "correlation_threshold": _check_positive("correlation_threshold", correlation_threshold),
        "correlation_window_s": _check_positive("correlation_window_s", correlation_window_s),
        "max_low_correlation_fraction": _check_positive(
            "max_low_correlation_fraction", max_low_correlation_fraction
        ),
        "hf_noise_threshold": _check_positive("hf_noise_threshold", hf_noise_threshold),
    }
    if params["correlation_threshold"] >= 1 or params["max_low_correlation_fraction"] >= 1:
        raise ValueError("correlation_threshold and max_low_correlation_fraction must be < 1.")
    if raw.n_times == 0:
        raise ValueError("Recording is empty (0 samples).")

    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if picks.size < 3:
        raise ValueError(f"Need at least 3 EEG channels to assess, found {picks.size}.")
    names = [raw.ch_names[i] for i in picks]
    x = raw.get_data(picks=picks)  # (n_channels, n_times), volts; a copy
    sfreq = raw.info["sfreq"]
    n_ch = len(names)

    metrics: dict[str, np.ndarray] = {}
    skipped: dict[str, str] = {}
    flags: dict[str, np.ndarray] = {}

    # nan
    finite = np.isfinite(x).all(axis=1)
    flags["nan"] = ~finite
    metrics["nan"] = (~np.isfinite(x)).mean(axis=1)
    x = np.where(np.isfinite(x), x, 0.0)  # keep other criteria computable

    # flat
    sd = x.std(axis=1)
    q75, q25 = np.percentile(x, [75, 25], axis=1)
    robust_sd = _IQR_TO_SD * (q75 - q25)
    flags["flat"] = (sd < params["flat_threshold"]) | (robust_sd < params["flat_threshold"])
    metrics["flat"] = np.minimum(sd, robust_sd)
    usable = finite & ~flags["flat"]

    # deviation (robust z of robust amplitude, computed over usable channels)
    z_dev = np.full(n_ch, np.nan)
    if usable.sum() >= 3:
        z_dev[usable] = _robust_z(robust_sd[usable])
    flags["deviation"] = np.abs(np.nan_to_num(z_dev)) > params["deviation_threshold"]
    metrics["deviation"] = z_dev

    # high-frequency noise and the low-passed copy used for correlation
    has_hf = sfreq > 2 * _HF_SPLIT_HZ and raw.info["lowpass"] > _HF_SPLIT_HZ
    x_low = x
    z_hf = np.full(n_ch, np.nan)
    if has_hf:
        x_low = mne.filter.filter_data(
            x, sfreq, None, _HF_SPLIT_HZ, method="fir", phase="zero", verbose="error"
        )
        hf = x - x_low

        def mad(a: np.ndarray) -> np.ndarray:
            return np.median(np.abs(a - np.median(a, axis=1, keepdims=True)), axis=1)

        ratio = np.full(n_ch, np.nan)
        lf_mad = mad(x_low)
        ok = usable & (lf_mad > 0)
        ratio[ok] = mad(hf)[ok] / lf_mad[ok]
        if ok.sum() >= 3:
            z_hf[ok] = _robust_z(ratio[ok])
        flags["hf_noise"] = np.nan_to_num(z_hf) > params["hf_noise_threshold"]
    else:
        skipped["hf_noise"] = (
            f"no content above {_HF_SPLIT_HZ:g} Hz (sfreq={sfreq:g} Hz, "
            f"lowpass={raw.info['lowpass']:g} Hz)"
        )
        flags["hf_noise"] = np.zeros(n_ch, bool)
    metrics["hf_noise"] = z_hf

    # correlation
    win = int(round(params["correlation_window_s"] * sfreq))
    n_win = x.shape[1] // win if win > 0 else 0
    frac_low = np.full(n_ch, np.nan)
    if n_win == 0:
        skipped["correlation"] = (
            f"recording ({x.shape[1] / sfreq:.3g} s) is shorter than one "
            f"{params['correlation_window_s']:g} s window"
        )
        flags["correlation"] = np.zeros(n_ch, bool)
    elif usable.sum() < 3:
        skipped["correlation"] = "fewer than 3 usable channels"
        flags["correlation"] = np.zeros(n_ch, bool)
    else:
        idx = np.flatnonzero(usable)
        low = np.zeros((idx.size, n_win), bool)
        for w in range(n_win):
            seg = x_low[idx, w * win : (w + 1) * win]
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.abs(np.corrcoef(seg))
            np.fill_diagonal(r, np.nan)
            max_r = np.nanmax(np.nan_to_num(r, nan=0.0), axis=1)
            low[:, w] = max_r < params["correlation_threshold"]
        frac_low[idx] = low.mean(axis=1)
        flags["correlation"] = np.nan_to_num(frac_low) > params["max_low_correlation_fraction"]
    metrics["correlation"] = frac_low

    reasons = {
        names[i]: tuple(c for c in CRITERIA if flags[c][i])
        for i in range(n_ch)
        if any(flags[c][i] for c in CRITERIA)
    }
    bad = tuple(n for n in names if n in reasons)
    return ChannelAssessment(
        channels=tuple(names),
        bad=bad,
        reasons=MappingProxyType(reasons),
        metrics=MappingProxyType(
            {c: MappingProxyType(dict(zip(names, map(float, metrics[c])))) for c in CRITERIA}
        ),
        skipped=MappingProxyType(skipped),
        previously_bad=tuple(n for n in names if n in raw.info["bads"]),
        params=MappingProxyType(params),
    )


def mark_bad_channels(
    inst: mne.io.BaseRaw | mne.BaseEpochs,
    channels: Sequence[str],
    *,
    replace: bool = False,
) -> mne.io.BaseRaw | mne.BaseEpochs:
    """Return a copy with ``channels`` listed in ``info["bads"]``.

    Parameters
    ----------
    inst : mne.io.BaseRaw or mne.BaseEpochs
        Data. Not modified.
    channels : sequence of str
        Channel names to mark, e.g. ``assessment.bad``. May be empty.
    replace : bool, default=False
        If False, add to the existing bads; if True, the result's bads are
        exactly ``channels``.

    Returns
    -------
    inst_marked : same type as ``inst``
        Copy whose ``info["bads"]`` is in recording order. Data are unchanged.

    Raises
    ------
    TypeError
        If ``inst`` is not MNE data, or ``channels`` is a string or holds non-strings.
    ValueError
        If a name is not a channel of ``inst`` or is duplicated.
    """
    if not isinstance(inst, (mne.io.BaseRaw, mne.BaseEpochs)):
        raise TypeError(f"Expected mne.io.BaseRaw or mne.Epochs, got {type(inst).__name__}.")
    if isinstance(channels, str) or not isinstance(channels, Sequence):
        raise TypeError("channels must be a list of channel names, e.g. ['T7'].")
    channels = list(channels)
    if not all(isinstance(c, str) for c in channels):
        raise TypeError("channel names must be strings.")
    if len(set(channels)) != len(channels):
        raise ValueError(f"Duplicate channel names: {channels}.")
    missing = [c for c in channels if c not in inst.ch_names]
    if missing:
        raise ValueError(f"Channel(s) not found: {missing}.")
    wanted = set(channels) if replace else set(channels) | set(inst.info["bads"])
    out = inst.copy()
    out.info["bads"] = [c for c in out.ch_names if c in wanted]
    return out


def _has_position(info: mne.Info, ch: str) -> bool:
    loc = info["chs"][info["ch_names"].index(ch)]["loc"][:3]
    return bool(np.all(np.isfinite(loc)) and np.any(loc != 0))


def interpolate_bad_channels(
    inst: mne.io.BaseRaw | mne.BaseEpochs,
    *,
    max_fraction: float = 0.2,
    reset_bads: bool = True,
) -> mne.io.BaseRaw | mne.BaseEpochs:
    """Replace bad EEG channels with spherical-spline interpolations.

    Parameters
    ----------
    inst : mne.io.BaseRaw or mne.BaseEpochs
        Data with bad EEG channels in ``info["bads"]`` and electrode
        positions for every EEG channel (e.g.
        ``raw.set_montage("colin27_1020")``). Not modified.
    max_fraction : float, default=0.2
        Refuse if more than this fraction of EEG channels is bad. A heuristic:
        interpolation from few good neighbours is unreliable, and MNE itself
        does not limit it (verified: it interpolates 12 of 16 channels).
    reset_bads : bool, default=True
        Remove interpolated channels from ``info["bads"]``. Non-EEG bads are
        always kept.

    Returns
    -------
    inst_interp : same type as ``inst``
        Copy in which only the bad EEG channels' data changed. With no bad
        EEG channels, an unchanged copy.

    Raises
    ------
    TypeError
        If ``inst`` is not MNE data.
    ValueError
        If any EEG channel lacks a position, too many channels are bad, or
        ``max_fraction`` is not in (0, 1).
    """
    if not isinstance(inst, (mne.io.BaseRaw, mne.BaseEpochs)):
        raise TypeError(f"Expected mne.io.BaseRaw or mne.Epochs, got {type(inst).__name__}.")
    max_fraction = _check_positive("max_fraction", max_fraction)
    if max_fraction >= 1:
        raise ValueError(f"max_fraction must be < 1, got {max_fraction}.")
    eeg = [inst.ch_names[i] for i in mne.pick_types(inst.info, eeg=True, exclude=[])]
    bad_eeg = [c for c in inst.info["bads"] if c in eeg]
    out = inst.copy()
    if not bad_eeg:
        return out
    if len(bad_eeg) / len(eeg) > max_fraction:
        raise ValueError(
            f"{len(bad_eeg)} of {len(eeg)} EEG channels are bad "
            f"({len(bad_eeg) / len(eeg):.0%} > max_fraction={max_fraction:.0%}); "
            "interpolation would rest on too few good channels."
        )
    no_pos = [c for c in eeg if not _has_position(inst.info, c)]
    if no_pos:
        raise ValueError(
            f"EEG channel(s) without electrode positions: {no_pos}. Set a montage "
            "first, e.g. raw.set_montage('colin27_1020') for standard 10-20 names."
        )
    out.load_data()
    # A bad non-EEG channel (e.g. EOG) is never interpolated, but MNE 1.13
    # still checks its position: with on_bad_position="raise" it refuses to
    # run, and with its default "warn" it silently drops the channel from
    # `bads` (verified; `exclude` does not prevent either). So hide non-EEG
    # bads during interpolation and restore them afterwards.
    non_eeg_bads = [c for c in inst.info["bads"] if c not in eeg]
    out.info["bads"] = bad_eeg
    out.interpolate_bads(
        reset_bads=reset_bads, mode="accurate", origin="auto",
        on_bad_position="raise", verbose="error",
    )
    kept = set(out.info["bads"]) | set(non_eeg_bads)
    out.info["bads"] = [c for c in out.ch_names if c in kept]
    return out
