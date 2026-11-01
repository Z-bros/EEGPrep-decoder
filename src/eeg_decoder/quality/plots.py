"""Quality plots that return figures and never show, block, or save.

Figures are built with matplotlib's object-oriented API
(``matplotlib.figure.Figure``), not ``pyplot``: no global figure state, no
GUI window, safe in scripts, notebooks, and headless servers. Display a
figure in a notebook by returning it, or save it with ``fig.savefig(path)``.

Colours follow a validated categorical palette (blue ``#2a78d6`` and orange
``#eb6834``: colour-vision-deficiency separation dE 24.7, contrast >= 3:1 on
the light background). Identity is never colour alone: bad channels also
carry a hatch pattern and a legend entry, and two-series lines are
direct-labelled as well as in a legend.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import mne
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from eeg_decoder.quality.metrics import EpochsQuality, RecordingQuality, _welch_psd_db

_SERIES = ("#2a78d6", "#eb6834")  # categorical slots 1-2
_INK, _INK_2, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
_METRICS = {  # metric: (axis label, scale, title)
    "robust_sd": ("Robust SD (µV)", 1e6, "Per-channel robust standard deviation"),
    "ptp": ("Peak-to-peak (µV)", 1e6, "Per-channel peak-to-peak amplitude"),
    "line_noise_db": ("Line noise (dB above neighbours)", 1.0, "Per-channel line noise"),
}


def _style(ax) -> None:
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK_2)
    ax.tick_params(colors=_INK_2, labelsize=9)
    ax.yaxis.label.set_color(_INK)
    ax.xaxis.label.set_color(_INK)
    ax.title.set_color(_INK)
    ax.grid(True, axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _mean_psd_db(raw: mne.io.BaseRaw, window_s: float):
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if picks.size == 0:
        raise ValueError("No good EEG channels to plot.")
    x = raw.get_data(picks=picks, reject_by_annotation="omit")
    if x.shape[1] < int(round(window_s * raw.info["sfreq"])):
        raise ValueError(f"Less than window_s={window_s} s of good data to estimate a PSD.")
    freqs, psd_db = _welch_psd_db(x, raw.info["sfreq"], window_s)
    with np.errstate(divide="ignore"):
        mean_db = 10 * np.log10(np.mean(10 ** (psd_db / 10), axis=0))  # average power, then dB
    return freqs, mean_db


def plot_psd_comparison(
    before: mne.io.BaseRaw,
    after: mne.io.BaseRaw,
    *,
    fmax: float | None = None,
    window_s: float = 2.0,
    labels: Sequence[str] = ("before", "after"),
) -> Figure:
    """Plot the mean EEG power spectrum of two versions of a recording.

    Parameters
    ----------
    before, after : mne.io.BaseRaw
        E.g. the raw recording and its preprocessed version. Not modified.
        Good EEG channels outside ``BAD*`` annotations are averaged (power
        averaged across channels, then converted to dB).
    fmax : float or None
        Upper frequency shown; defaults to the lower of the two Nyquist frequencies.
    window_s : float, default=2.0
        Welch window length (s); frequency resolution is ``1 / window_s`` Hz.
    labels : pair of str
        Names for the two spectra.

    Returns
    -------
    matplotlib.figure.Figure
        Not shown and not saved.

    Raises
    ------
    TypeError
        If an input is not continuous MNE data.
    ValueError
        If there are no good EEG channels, too little data, or invalid
        ``fmax``/``window_s``/``labels``.
    """
    for name, inst in (("before", before), ("after", after)):
        if isinstance(inst, mne.BaseEpochs) or not isinstance(inst, mne.io.BaseRaw):
            raise TypeError(f"{name} must be mne.io.BaseRaw, got {type(inst).__name__}.")
    if isinstance(labels, str) or len(labels) != 2:
        raise ValueError("labels must be two strings.")
    if isinstance(window_s, bool) or not isinstance(window_s, (int, float)) or not (
        math.isfinite(window_s) and window_s > 0
    ):
        raise ValueError(f"window_s must be a positive number, got {window_s!r}.")
    top = min(before.info["sfreq"], after.info["sfreq"]) / 2
    if fmax is None:
        fmax = top
    elif isinstance(fmax, bool) or not isinstance(fmax, (int, float)) or not 0 < fmax <= top:
        raise ValueError(f"fmax must be in (0, {top:g}] Hz, got {fmax!r}.")

    fig = Figure(figsize=(7, 4), layout="constrained", facecolor=_SURFACE)
    ax = fig.add_subplot()
    _style(ax)
    ends = []
    for (label, inst), color in zip(zip(labels, (before, after)), _SERIES):
        freqs, db = _mean_psd_db(inst, window_s)
        # Skip the DC and Nyquist bins: Welch edge bins, not signal.
        keep = (freqs > 0) & (freqs <= fmax) & (freqs < inst.info["sfreq"] / 2)
        ax.plot(freqs[keep], db[keep], color=color, linewidth=2, label=label)
        ends.append((label, freqs[keep][-1], db[keep][-1]))
    # Direct labels at the line ends; when the two ends are close (e.g. the
    # spectra coincide at high frequencies), push them apart vertically.
    lo, hi = ax.get_ylim()
    close = abs(ends[0][2] - ends[1][2]) < 0.06 * (hi - lo)
    upper = 0 if ends[0][2] >= ends[1][2] else 1
    for i, (label, x_end, y_end) in enumerate(ends):
        dy = (7 if i == upper else -7) if close else 0
        ax.annotate(label, (x_end, y_end), xytext=(4, dy), textcoords="offset points",
                    va="center", fontsize=9, color=_INK)
    ax.set_xlim(0, fmax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB re 1 µV²/Hz)")
    ax.set_title("Mean EEG power spectrum (good channels, non-BAD data)", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK)
    return fig


def plot_channel_quality(quality: RecordingQuality, metric: str = "robust_sd") -> Figure:
    """Bar chart of one per-channel metric, with bad channels marked.

    Parameters
    ----------
    quality : RecordingQuality
        From :func:`~eeg_decoder.quality.recording_quality`.
    metric : {"robust_sd", "ptp", "line_noise_db"}, default="robust_sd"
        Metric to plot (amplitudes shown in µV).

    Returns
    -------
    matplotlib.figure.Figure
        Bad channels drawn in orange with a hatch pattern; the dashed line is
        the median of the good channels. Not shown and not saved.

    Raises
    ------
    TypeError
        If ``quality`` is not a :class:`RecordingQuality`.
    ValueError
        If ``metric`` is unknown or was not computed.
    """
    if not isinstance(quality, RecordingQuality):
        raise TypeError(f"Expected RecordingQuality, got {type(quality).__name__}.")
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {tuple(_METRICS)}, got {metric!r}.")
    if metric not in quality.channel_metrics:
        raise ValueError(f"{metric!r} was not computed: {quality.skipped.get(metric, 'unknown')}.")
    ylabel, scale, title = _METRICS[metric]
    names = list(quality.eeg_channels)
    values = np.array([quality.channel_metrics[metric][n] for n in names]) * scale
    is_bad = np.array([n in quality.bad_channels for n in names])

    fig = Figure(figsize=(max(5, 0.35 * len(names) + 1.5), 3.6), layout="constrained",
                 facecolor=_SURFACE)
    ax = fig.add_subplot()
    _style(ax)
    x = np.arange(len(names))
    ax.bar(x[~is_bad], values[~is_bad], width=0.7, color=_SERIES[0], label="good")
    if is_bad.any():
        ax.bar(x[is_bad], values[is_bad], width=0.7, color=_SERIES[1], hatch="///",
               edgecolor=_SURFACE, linewidth=0, label="bad (info['bads'])")
    good_vals = values[~is_bad & np.isfinite(values)]
    if good_vals.size:
        ax.axhline(np.median(good_vals), color=_INK_2, linewidth=1, linestyle="--",
                   label="median of good channels")
    ax.set_xticks(x, names, rotation=90 if len(names) > 20 else 0, fontsize=8)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK)
    return fig


def plot_rejection_summary(quality: EpochsQuality, *, max_reasons: int = 10) -> Figure:
    """Horizontal bars of why epochs were dropped, with kept/dropped totals in the title.

    Parameters
    ----------
    quality : EpochsQuality
        From :func:`~eeg_decoder.quality.epochs_quality`.
    max_reasons : int, default=10
        Show the most frequent reasons; the rest are folded into "other".

    Returns
    -------
    matplotlib.figure.Figure
        Not shown and not saved. With no dropped epochs the axes say so.

    Raises
    ------
    TypeError
        If ``quality`` is not an :class:`EpochsQuality`.
    ValueError
        If ``max_reasons`` is not a positive int.
    """
    if not isinstance(quality, EpochsQuality):
        raise TypeError(f"Expected EpochsQuality, got {type(quality).__name__}.")
    if isinstance(max_reasons, bool) or not isinstance(max_reasons, int) or max_reasons < 1:
        raise ValueError(f"max_reasons must be a positive int, got {max_reasons!r}.")
    items = list(quality.reasons.items())
    if len(items) > max_reasons:
        rest = sum(n for _, n in items[max_reasons - 1 :])
        items = items[: max_reasons - 1] + [("other", rest)]

    fig = Figure(figsize=(6, 1.2 + 0.35 * max(len(items), 1)), layout="constrained",
                 facecolor=_SURFACE)
    ax = fig.add_subplot()
    _style(ax)
    ax.grid(True, axis="x", color=_GRID, linewidth=0.8)
    ax.grid(False, axis="y")
    ax.set_title(f"{quality.n_kept} of {quality.n_events} epochs kept, "
                 f"{quality.n_dropped} dropped ({quality.drop_fraction:.0%})", fontsize=11)
    if not items:
        ax.text(0.5, 0.5, "No epochs dropped", ha="center", va="center",
                transform=ax.transAxes, color=_INK_2)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig
    labels = [r for r, _ in items][::-1]
    counts = [n for _, n in items][::-1]
    bars = ax.barh(labels, counts, height=0.6, color=_SERIES[0])
    ax.bar_label(bars, padding=3, fontsize=9, color=_INK)
    ax.set_xlabel("Dropped epochs listing this reason")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(0, max(counts) * 1.15)
    return fig
