"""Check that the synthetic fixtures really contain what they claim.

These tests guard the *test infrastructure*: later phases (filtering, PSD,
CSP) rely on these known spectral properties as ground truth.
"""

import numpy as np

from conftest import ALPHA_HZ, LINE_HZ, make_continuous, make_epochs


def _peak_freqs(x: np.ndarray, sfreq: float, n_peaks: int = 2) -> np.ndarray:
    """Frequencies of the strongest rFFT bins of a 1-D signal."""
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, 1 / sfreq)
    return np.sort(freqs[np.argsort(spec)[-n_peaks:]])


def test_continuous_shape_units_and_finiteness(continuous_eeg):
    d = continuous_eeg
    assert d.data.shape == (len(d.ch_names), int(10 * d.sfreq))
    assert np.isfinite(d.data).all()
    # Volts: EEG is tens of microvolts, so values must be ~1e-5, not ~10.
    assert 1e-6 < np.abs(d.data).max() < 1e-3


def test_continuous_spectral_peaks_are_10_and_50_hz(continuous_eeg):
    d = continuous_eeg
    for ch in range(d.data.shape[0]):
        peaks = _peak_freqs(d.data[ch], d.sfreq)
        np.testing.assert_allclose(peaks, [ALPHA_HZ, LINE_HZ], atol=0.2)


def test_fixed_seed_is_reproducible_and_seed_matters():
    a, b, c = make_continuous(seed=1), make_continuous(seed=1), make_continuous(seed=2)
    np.testing.assert_array_equal(a.data, b.data)
    assert not np.array_equal(a.data, c.data)


def test_epochs_shape_labels_and_class_separation(epoched_eeg):
    e = epoched_eeg
    n_ep, n_ch, n_samp = e.data.shape
    assert (n_ch, n_samp) == (len(e.ch_names), int(2 * e.sfreq))
    assert e.labels.shape == (n_ep,)
    assert set(np.unique(e.labels)) == {0, 1}
    assert (e.labels == 0).sum() == (e.labels == 1).sum()  # balanced
    # Class 1 was built with 2x the alpha amplitude -> clearly higher variance.
    var = e.data.var(axis=-1).mean(axis=-1)
    assert var[e.labels == 1].mean() > 2 * var[e.labels == 0].mean()


def test_epochs_reproducible():
    a, b = make_epochs(seed=3), make_epochs(seed=3)
    np.testing.assert_array_equal(a.data, b.data)
    np.testing.assert_array_equal(a.labels, b.labels)
