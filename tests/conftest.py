"""Synthetic EEG fixtures with known spectral content.

All signals are generated from explicit seeds, are expressed in **volts**
(SI, as in MNE-Python), and follow the package axis conventions:

- continuous: ``(n_channels, n_samples)``
- epoched:    ``(n_epochs, n_channels, n_samples)``

No real EEG dataset is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

SFREQ = 250.0
CH_NAMES = ("Fz", "Cz", "Pz", "C3", "C4", "O1", "O2", "Oz")

ALPHA_HZ = 10.0
LINE_HZ = 50.0
ALPHA_AMP_V = 20e-6  # 20 uV peak
LINE_AMP_V = 5e-6  # 5 uV peak
NOISE_STD_V = 2e-6  # 2 uV rms white noise

FIXTURE_SEED = 1234


@dataclass(frozen=True)
class SyntheticContinuous:
    """Continuous recording plus the ground truth used to build it."""

    data: np.ndarray  # (n_channels, n_samples), volts
    sfreq: float
    ch_names: tuple[str, ...]
    alpha_hz: float
    line_hz: float


@dataclass(frozen=True)
class SyntheticEpochs:
    """Two-class epoched data; class 1 has stronger alpha than class 0."""

    data: np.ndarray  # (n_epochs, n_channels, n_samples), volts
    labels: np.ndarray  # (n_epochs,), values in {0, 1}
    sfreq: float
    ch_names: tuple[str, ...]
    alpha_hz: float
    alpha_amp_by_class_v: tuple[float, float]


def make_continuous(seed: int = FIXTURE_SEED, duration_s: float = 10.0) -> SyntheticContinuous:
    """10 Hz alpha + 50 Hz line noise + white noise, per channel random phase."""
    rng = np.random.default_rng(seed)
    n_ch, n_samp = len(CH_NAMES), int(round(duration_s * SFREQ))
    t = np.arange(n_samp) / SFREQ
    phase_a = rng.uniform(0, 2 * np.pi, size=(n_ch, 1))
    phase_l = rng.uniform(0, 2 * np.pi, size=(n_ch, 1))
    data = (
        ALPHA_AMP_V * np.sin(2 * np.pi * ALPHA_HZ * t + phase_a)
        + LINE_AMP_V * np.sin(2 * np.pi * LINE_HZ * t + phase_l)
        + rng.normal(0.0, NOISE_STD_V, size=(n_ch, n_samp))
    )
    return SyntheticContinuous(data, SFREQ, CH_NAMES, ALPHA_HZ, LINE_HZ)


def make_epochs(
    seed: int = FIXTURE_SEED, n_epochs: int = 40, epoch_s: float = 2.0
) -> SyntheticEpochs:
    """Balanced two-class epochs; alpha amplitude 10 uV (class 0) vs 20 uV (class 1)."""
    rng = np.random.default_rng(seed)
    n_ch, n_samp = len(CH_NAMES), int(round(epoch_s * SFREQ))
    t = np.arange(n_samp) / SFREQ
    amps = (10e-6, 20e-6)
    labels = np.tile([0, 1], n_epochs // 2)
    rng.shuffle(labels)
    phase = rng.uniform(0, 2 * np.pi, size=(n_epochs, n_ch, 1))
    amp = np.asarray(amps)[labels][:, None, None]
    data = amp * np.sin(2 * np.pi * ALPHA_HZ * t + phase) + rng.normal(
        0.0, NOISE_STD_V, size=(n_epochs, n_ch, n_samp)
    )
    return SyntheticEpochs(data, labels, SFREQ, CH_NAMES, ALPHA_HZ, amps)


@pytest.fixture
def continuous_eeg() -> SyntheticContinuous:
    return make_continuous()


@pytest.fixture
def epoched_eeg() -> SyntheticEpochs:
    return make_epochs()


@pytest.fixture
def continuous_raw(continuous_eeg):
    """The continuous fixture wrapped as an ``mne.io.RawArray`` (volts)."""
    from eeg_decoder.io import to_raw

    d = continuous_eeg
    return to_raw(d.data, d.sfreq, d.ch_names)
