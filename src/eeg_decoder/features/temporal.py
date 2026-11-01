"""Time-domain statistics and Hjorth parameters, per epoch.

Each value uses only one epoch's own samples (leakage-free).

Hjorth parameters (Hjorth, 1970), with derivatives taken as sample
differences scaled by ``sfreq`` so units are physical:

* activity   = var(x)                          [V^2]
* mobility   = sqrt(var(x') / var(x))          [rad/s]; ~ 2*pi*f for a sine at f Hz
* complexity = mobility(x') / mobility(x)      [unitless]; 1 for a pure sine

Mobility and complexity are dominated by high-frequency content, so broadband
noise inflates them strongly (verified: a 10 Hz sine plus 10 % white noise gave
mobility 78.7 instead of 62.8 rad/s and complexity 3.4 instead of ~1). Low-pass
the data first. A channel that is constant within an epoch has zero variance;
its mobility and complexity are NaN.
"""

from __future__ import annotations

from collections.abc import Sequence

import mne
import numpy as np
from scipy import stats as sps

from eeg_decoder.features._array import FeatureArray
from eeg_decoder.features._checks import check_epochs

_STATS = {
    "mean": ("V", lambda x: x.mean(axis=-1)),
    "var": ("V^2", lambda x: x.var(axis=-1)),
    "std": ("V", lambda x: x.std(axis=-1)),
    "ptp": ("V", lambda x: np.ptp(x, axis=-1)),
    "rms": ("V", lambda x: np.sqrt((x ** 2).mean(axis=-1))),
    "skew": ("unitless", lambda x: sps.skew(x, axis=-1, bias=True)),
    "kurtosis": ("unitless", lambda x: sps.kurtosis(x, axis=-1, fisher=True, bias=True)),
}


def time_stats(
    epochs: mne.BaseEpochs,
    stats: Sequence[str] = ("mean", "var", "std", "ptp", "rms", "skew", "kurtosis"),
    *,
    include_bads: bool = False,
) -> FeatureArray:
    """Simple amplitude statistics of every EEG channel in every epoch.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epoched data. Not modified.
    stats : sequence of str
        Any of ``mean``, ``var``, ``std``, ``ptp``, ``rms`` (volts or V^2),
        ``skew``, ``kurtosis`` (Fisher's excess kurtosis: 0 for a Gaussian).
        Population (biased) estimators throughout.
    include_bads : bool, default=False
        Include channels marked bad.

    Returns
    -------
    FeatureArray
        dims ``("epoch", "channel", "stat")``. Because units differ between
        statistics, ``units`` lists them per statistic, e.g. ``"var=V^2, ..."``.

    Raises
    ------
    TypeError, ValueError
        On invalid epochs, or unknown, duplicate, or empty ``stats``.
    """
    if isinstance(stats, str) or not isinstance(stats, Sequence):
        raise TypeError("stats must be a list of statistic names.")
    stats = list(stats)
    if not stats:
        raise ValueError("stats must not be empty.")
    unknown = [s for s in stats if s not in _STATS]
    if unknown:
        raise ValueError(f"Unknown statistic(s) {unknown}; choose from {sorted(_STATS)}.")
    if len(set(stats)) != len(stats):
        raise ValueError(f"Duplicate statistics: {stats}.")
    data, names, _ = check_epochs(epochs, include_bads=include_bads)
    values = np.stack([_STATS[s][1](data) for s in stats], axis=-1)
    units = ", ".join(f"{s}={_STATS[s][0]}" for s in stats)
    return FeatureArray._from_epochs(
        epochs, values, ("epoch", "channel", "stat"), {"channel": names, "stat": stats}, units
    )


def hjorth(epochs: mne.BaseEpochs, *, include_bads: bool = False) -> FeatureArray:
    """Hjorth activity, mobility, and complexity of every EEG channel in every epoch.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epoched data, ideally low-pass filtered (see module notes). Not modified.
    include_bads : bool, default=False
        Include channels marked bad.

    Returns
    -------
    FeatureArray
        dims ``("epoch", "channel", "parameter")`` with parameters
        ``activity`` (V^2), ``mobility`` (rad/s), ``complexity`` (unitless).

    Raises
    ------
    TypeError, ValueError
        On invalid epochs, or epochs shorter than 3 samples.
    """
    data, names, sfreq = check_epochs(epochs, include_bads=include_bads)
    if data.shape[-1] < 3:
        raise ValueError("Hjorth parameters need at least 3 samples per epoch.")
    d1 = np.diff(data, axis=-1) * sfreq
    d2 = np.diff(d1, axis=-1) * sfreq
    v0, v1, v2 = data.var(axis=-1), d1.var(axis=-1), d2.var(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mobility = np.sqrt(v1 / v0)
        complexity = np.sqrt(v2 / v1) / mobility
    values = np.stack([v0, mobility, complexity], axis=-1)
    return FeatureArray._from_epochs(
        epochs, values, ("epoch", "channel", "parameter"),
        {"channel": names, "parameter": ["activity", "mobility", "complexity"]},
        "activity=V^2, mobility=rad/s, complexity=unitless",
    )
