"""Quantitative, machine-readable data-quality summaries.

Everything here is **descriptive**: metrics are reported, never turned into
pass/fail verdicts, because acceptable values depend on the montage, the
paradigm, and the lab. Use them to compare recordings, spot outliers, and
document data quality alongside results.

* :func:`recording_quality` summarises a continuous recording: bad channels,
  time covered by ``BAD_*`` annotations, and per-channel amplitude and
  line-noise metrics computed on the good (non-``BAD``) data only.
* :func:`epochs_quality` summarises epoching and rejection from
  ``epochs.drop_log``: how many events were kept, dropped, and why.

Both return frozen dataclasses with ``to_dict()`` (JSON-serializable, NaN
becomes ``None``) and ``summary()`` (human-readable).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import mne
import numpy as np
from scipy.signal import welch

_IQR_TO_SD = 0.7413
# Line-noise ratio: power within +-0.5 Hz of the line frequency versus the
# median power 2-5 Hz away on both sides, from a Welch PSD with 2 s windows.
_LINE_HALF_WIDTH_HZ = 0.5
_REF_NEAR_HZ, _REF_FAR_HZ = 2.0, 5.0
_PSD_WINDOW_S = 2.0
# drop_log entries that are bookkeeping, not rejections.
_NOT_EVENTS = {"IGNORED"}


def _clean(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _welch_psd_db(x: np.ndarray, sfreq: float, window_s: float = _PSD_WINDOW_S):
    """Welch PSD (Hann, 50 % overlap) in dB re 1 uV^2/Hz; x in volts, last axis time."""
    nper = int(round(window_s * sfreq))
    freqs, psd = welch(x, fs=sfreq, window="hann", nperseg=nper, noverlap=nper // 2, axis=-1)
    with np.errstate(divide="ignore"):
        return freqs, 10 * np.log10(psd / 1e-12)


@dataclass(frozen=True)
class RecordingQuality:
    """Descriptive quality metrics of one continuous recording.

    Attributes
    ----------
    sfreq, duration_s : float
        Sampling rate (Hz) and length (s) of the recording.
    highpass, lowpass : float
        Filter edges recorded in ``info``.
    custom_ref_applied : bool
        Whether the reference was changed from the recording reference.
    eeg_channels : tuple of str
        All EEG channels, in recording order.
    bad_channels : tuple of str
        EEG channels in ``info["bads"]``.
    bad_channel_fraction : float
        ``len(bad_channels) / len(eeg_channels)``.
    bad_span_s, bad_span_fraction : float
        Time covered by ``BAD*`` annotations (union, clipped to the data).
    annotation_s : mapping of str to float
        Seconds covered per annotation description (all descriptions).
    channel_metrics : mapping of str to mapping of str to float
        ``"robust_sd"`` and ``"ptp"`` (volts) and, if computed,
        ``"line_noise_db"`` per EEG channel, on non-``BAD`` data.
    line_freq : float or None
        Line frequency used for ``"line_noise_db"``.
    skipped : mapping of str to str
        Metrics not computed, with the reason.
    """

    sfreq: float
    duration_s: float
    highpass: float
    lowpass: float
    custom_ref_applied: bool
    eeg_channels: tuple[str, ...]
    bad_channels: tuple[str, ...]
    bad_channel_fraction: float
    bad_span_s: float
    bad_span_fraction: float
    annotation_s: Mapping[str, float]
    channel_metrics: Mapping[str, Mapping[str, float]]
    line_freq: float | None
    skipped: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable copy (NaN becomes ``None``)."""
        return {
            "sfreq": self.sfreq,
            "duration_s": self.duration_s,
            "highpass": self.highpass,
            "lowpass": self.lowpass,
            "custom_ref_applied": self.custom_ref_applied,
            "eeg_channels": list(self.eeg_channels),
            "bad_channels": list(self.bad_channels),
            "bad_channel_fraction": self.bad_channel_fraction,
            "bad_span_s": self.bad_span_s,
            "bad_span_fraction": self.bad_span_fraction,
            "annotation_s": dict(self.annotation_s),
            "channel_metrics": {
                m: {ch: _clean(v) for ch, v in vals.items()} for m, vals in self.channel_metrics.items()
            },
            "line_freq": self.line_freq,
            "skipped": dict(self.skipped),
        }

    def summary(self) -> str:
        """Human-readable overview."""
        lines = [
            f"{self.duration_s:.1f} s at {self.sfreq:g} Hz, {len(self.eeg_channels)} EEG channels "
            f"(filter {self.highpass:g}-{self.lowpass:g} Hz, "
            f"{'custom' if self.custom_ref_applied else 'recording'} reference)",
            f"bad channels: {len(self.bad_channels)} ({self.bad_channel_fraction:.0%})"
            + (f" {list(self.bad_channels)}" if self.bad_channels else ""),
            f"BAD-annotated time: {self.bad_span_s:.2f} s ({self.bad_span_fraction:.1%})",
        ]
        sd = [v for ch, v in self.channel_metrics["robust_sd"].items()
              if ch not in self.bad_channels and math.isfinite(v)]
        if sd:
            lines.append(
                f"robust SD of good channels: median {np.median(sd) * 1e6:.1f} uV "
                f"(range {min(sd) * 1e6:.1f}-{max(sd) * 1e6:.1f} uV)"
            )
        ln = self.channel_metrics.get("line_noise_db", {})
        good_ln = [v for ch, v in ln.items() if ch not in self.bad_channels and math.isfinite(v)]
        if good_ln:
            lines.append(f"line noise at {self.line_freq:g} Hz: median {np.median(good_ln):.1f} dB "
                         "above neighbouring frequencies")
        lines += [f"(skipped {k}: {why})" for k, why in self.skipped.items()]
        return "\n".join(lines)


def recording_quality(
    raw: mne.io.BaseRaw,
    *,
    line_freq: float | None = None,
) -> RecordingQuality:
    """Summarise the quality of a continuous recording.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous recording with at least one EEG channel. Not modified.
    line_freq : float or None
        Mains frequency (e.g. 50 or 60 Hz) for the line-noise metric. Never
        assumed: ``None`` skips the metric.

    Returns
    -------
    RecordingQuality
        Descriptive metrics; see the class attributes.

    Raises
    ------
    TypeError
        If ``raw`` is not continuous MNE data or ``line_freq`` is not a number.
    ValueError
        If the recording is empty, has no EEG channel, or ``line_freq`` is
        not in (0, Nyquist).

    Notes
    -----
    Channel metrics use only samples outside ``BAD*`` annotations (MNE's
    ``reject_by_annotation``), so marked artifacts do not inflate them. The
    line-noise metric needs ``line_freq + 5`` Hz below Nyquist and at least
    2 s of good data; otherwise it is skipped with the reason.
    """
    if isinstance(raw, mne.BaseEpochs) or not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(
            f"recording_quality expects continuous data (mne.io.BaseRaw), got {type(raw).__name__}."
        )
    if raw.n_times == 0:
        raise ValueError("Recording is empty (0 samples).")
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if picks.size == 0:
        raise ValueError("No EEG channels found.")
    sfreq = float(raw.info["sfreq"])
    nyquist = sfreq / 2
    if line_freq is not None:
        if isinstance(line_freq, bool) or not isinstance(line_freq, (int, float, np.number)):
            raise TypeError(f"line_freq must be a number or None, got {type(line_freq).__name__}.")
        if not math.isfinite(line_freq) or not 0 < line_freq < nyquist:
            raise ValueError(f"line_freq must be in (0, {nyquist:g}) Hz, got {line_freq}.")
        line_freq = float(line_freq)

    names = tuple(raw.ch_names[i] for i in picks)
    bads = tuple(n for n in names if n in raw.info["bads"])

    # BAD spans: MNE's own masking, so the union and clipping match epoching.
    masked = raw.get_data(picks=[picks[0]], reject_by_annotation="NaN")[0]
    bad_mask = np.isnan(masked) & np.isfinite(raw.get_data(picks=[picks[0]])[0])
    bad_span_s = float(bad_mask.sum() / sfreq)

    annotation_s: dict[str, float] = {}
    for desc in sorted(set(raw.annotations.description)):
        covered = np.zeros(raw.n_times, bool)
        for onset, dur, d in zip(raw.annotations.onset, raw.annotations.duration,
                                 raw.annotations.description):
            if d != desc:
                continue
            start = int(round((onset - raw.first_time) * sfreq))
            stop = int(round((onset - raw.first_time + dur) * sfreq))
            covered[max(start, 0) : min(stop, raw.n_times)] = True  # zero duration -> 0 s
        annotation_s[desc] = float(covered.sum() / sfreq)

    good = raw.get_data(picks=picks, reject_by_annotation="omit")
    skipped: dict[str, str] = {}
    nan = np.full(len(names), np.nan)
    if good.shape[1] == 0:
        robust_sd = ptp = nan
        skipped["channel_metrics"] = "no data outside BAD annotations"
    else:
        q75, q25 = np.percentile(good, [75, 25], axis=1)
        robust_sd = _IQR_TO_SD * (q75 - q25)
        ptp = np.ptp(good, axis=1)
    metrics: dict[str, Mapping[str, float]] = {
        "robust_sd": MappingProxyType(dict(zip(names, map(float, robust_sd)))),
        "ptp": MappingProxyType(dict(zip(names, map(float, ptp)))),
    }

    if line_freq is None:
        skipped["line_noise_db"] = "line_freq not given"
    elif line_freq + _REF_FAR_HZ >= nyquist:
        skipped["line_noise_db"] = (
            f"reference band up to {line_freq + _REF_FAR_HZ:g} Hz exceeds Nyquist ({nyquist:g} Hz)"
        )
    elif good.shape[1] < int(round(_PSD_WINDOW_S * sfreq)):
        skipped["line_noise_db"] = f"less than {_PSD_WINDOW_S:g} s of good data"
    else:
        freqs, psd_db = _welch_psd_db(good, sfreq)
        power = 10 ** (psd_db / 10)
        peak = np.abs(freqs - line_freq) <= _LINE_HALF_WIDTH_HZ
        dist = np.abs(freqs - line_freq)
        ref = (dist >= _REF_NEAR_HZ) & (dist <= _REF_FAR_HZ)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_db = 10 * np.log10(power[:, peak].mean(axis=1) / np.median(power[:, ref], axis=1))
        metrics["line_noise_db"] = MappingProxyType(dict(zip(names, map(float, ratio_db))))

    return RecordingQuality(
        sfreq=sfreq,
        duration_s=raw.n_times / sfreq,
        highpass=float(raw.info["highpass"]),
        lowpass=float(raw.info["lowpass"]),
        custom_ref_applied=bool(raw.info["custom_ref_applied"]),
        eeg_channels=names,
        bad_channels=bads,
        bad_channel_fraction=len(bads) / len(names),
        bad_span_s=bad_span_s,
        bad_span_fraction=bad_span_s / (raw.n_times / sfreq),
        annotation_s=MappingProxyType(annotation_s),
        channel_metrics=MappingProxyType(metrics),
        line_freq=line_freq,
        skipped=MappingProxyType(skipped),
    )


@dataclass(frozen=True)
class EpochsQuality:
    """What happened to each event during epoching and rejection.

    Attributes
    ----------
    n_events : int
        Events that were candidates for epochs (excludes ``IGNORED`` events,
        whose code was not in ``event_id``).
    n_kept, n_dropped : int
        Epochs remaining and dropped.
    drop_fraction : float
        ``n_dropped / n_events``.
    reasons : mapping of str to int
        For each reason in ``drop_log`` (a channel name for threshold
        rejection, an annotation description such as ``BAD_blink``, or MNE's
        ``NO_DATA``/``TOO_SHORT``/``USER``), the number of dropped epochs that
        list it. An epoch can list several reasons.
    kept_per_condition : mapping of str to int
        Remaining epochs per ``event_id`` name.
    """

    n_events: int
    n_kept: int
    n_dropped: int
    drop_fraction: float
    reasons: Mapping[str, int]
    kept_per_condition: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable copy."""
        return {
            "n_events": self.n_events,
            "n_kept": self.n_kept,
            "n_dropped": self.n_dropped,
            "drop_fraction": self.drop_fraction,
            "reasons": dict(self.reasons),
            "kept_per_condition": dict(self.kept_per_condition),
        }

    def summary(self) -> str:
        """Human-readable overview."""
        lines = [f"{self.n_kept} of {self.n_events} epochs kept, {self.n_dropped} dropped "
                 f"({self.drop_fraction:.1%})"]
        lines += [f"  {name}: {n} kept" for name, n in self.kept_per_condition.items()]
        lines += [f"  dropped for {r}: {n}" for r, n in self.reasons.items()]
        return "\n".join(lines)


def epochs_quality(epochs: mne.BaseEpochs) -> EpochsQuality:
    """Summarise kept and dropped epochs from ``epochs.drop_log``.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epochs, e.g. after :func:`~eeg_decoder.artifacts.reject_epochs`. Not modified.

    Returns
    -------
    EpochsQuality
        Counts and reasons; see the class attributes.

    Raises
    ------
    TypeError
        If ``epochs`` is not MNE Epochs.

    Notes
    -----
    MNE does not keep the event codes of dropped epochs, so per-condition
    counts are available only for the kept ones.
    """
    if not isinstance(epochs, mne.BaseEpochs):
        raise TypeError(f"epochs_quality expects mne.Epochs, got {type(epochs).__name__}.")
    logs = [entry for entry in epochs.drop_log if not set(entry) & _NOT_EVENTS]
    n_events = len(logs)
    n_dropped = sum(1 for entry in logs if entry)
    reasons = Counter(r for entry in logs for r in set(entry))
    codes = epochs.events[:, 2] if len(epochs.events) else np.array([], int)
    kept = {name: int(np.sum(codes == code)) for name, code in epochs.event_id.items()}
    return EpochsQuality(
        n_events=n_events,
        n_kept=n_events - n_dropped,
        n_dropped=n_dropped,
        drop_fraction=n_dropped / n_events if n_events else 0.0,
        reasons=MappingProxyType(dict(reasons.most_common())),
        kept_per_condition=MappingProxyType(kept),
    )
