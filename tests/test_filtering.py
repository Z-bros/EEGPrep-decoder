"""Band-pass and notch filters, validated on signals with known spectra."""

import mne
import numpy as np
import pytest

from eeg_decoder.io import to_raw
from eeg_decoder.preprocessing import bandpass, notch

SFREQ = 250.0
DURATION_S = 20.0  # 0.05 Hz FFT resolution; every test tone falls on a bin


def _tones(freqs, amp=10e-6, n_ch=2, dc=0.0):
    """Equal-amplitude sinusoids at `freqs` plus a DC offset, no noise."""
    t = np.arange(int(DURATION_S * SFREQ)) / SFREQ
    sig = dc + sum(amp * np.sin(2 * np.pi * f * t) for f in freqs)
    return to_raw(np.tile(sig, (n_ch, 1)), SFREQ, [f"ch{i}" for i in range(n_ch)])


def _amplitude(raw, freq):
    """Single-tone amplitude from the FFT, ignoring 2 s at each edge."""
    edge = int(2 * SFREQ)
    x = raw.get_data()[0, edge:-edge]
    spec = np.abs(np.fft.rfft(x)) * 2 / x.size
    freqs = np.fft.rfftfreq(x.size, 1 / SFREQ)
    return spec[np.argmin(np.abs(freqs - freq))]


def _gain(before, after, freq):
    return _amplitude(after, freq) / _amplitude(before, freq)


# --- frequency response -----------------------------------------------------


def test_bandpass_keeps_passband_and_removes_stopbands():
    # 1-40 Hz with MNE "auto" transitions: high-pass transition 0-1 Hz,
    # low-pass transition 40-50 Hz. Hamming window -> ~53 dB stopband.
    raw = _tones([0.2, 10.0, 20.0, 60.0], dc=50e-6)
    out = bandpass(raw, 1.0, 40.0)
    assert _gain(raw, out, 10.0) == pytest.approx(1.0, abs=0.01)
    assert _gain(raw, out, 20.0) == pytest.approx(1.0, abs=0.01)
    assert _gain(raw, out, 0.0) < 1e-3  # DC offset: stopband, > 60 dB
    assert _gain(raw, out, 60.0) < 3e-3  # beyond transition: stopband, > 50 dB
    # 0.2 Hz sits INSIDE the 0-1 Hz transition band: only partly attenuated.
    assert 0.01 < _gain(raw, out, 0.2) < 0.1  # ~ -22 dB


def test_one_sided_highpass_and_lowpass():
    raw = _tones([10.0, 80.0], dc=50e-6)
    hp = bandpass(raw, 1.0, None)
    assert _gain(raw, hp, 0.0) < 1e-3
    assert _gain(raw, hp, 80.0) == pytest.approx(1, abs=0.01)
    lp = bandpass(raw, None, 40.0)
    assert _gain(raw, lp, 80.0) < 3e-3
    assert _gain(raw, lp, 0.0) == pytest.approx(1, abs=0.01)


def test_notch_removes_line_noise_and_keeps_neighbours():
    raw = _tones([10.0, 45.0, 50.0, 100.0])
    out = notch(raw, [50.0, 100.0])
    assert _gain(raw, out, 50.0) < 0.01
    assert _gain(raw, out, 100.0) < 0.01
    assert _gain(raw, out, 10.0) == pytest.approx(1.0, abs=0.01)
    assert _gain(raw, out, 45.0) == pytest.approx(1.0, abs=0.01)  # 5 Hz away: untouched


def test_zero_phase_no_time_shift():
    raw = _tones([10.0])
    out = bandpass(raw, 1.0, 40.0)
    edge = int(2 * SFREQ)
    x, y = raw.get_data()[0, edge:-edge], out.get_data()[0, edge:-edge]
    lags = np.arange(-10, 11)
    xcorr = [np.dot(x, np.roll(y, k)) for k in lags]
    assert lags[int(np.argmax(xcorr))] == 0


# --- parity with the reference implementation ------------------------------


def test_matches_mne_directly(continuous_raw):
    ours = bandpass(continuous_raw, 1.0, 40.0).get_data()
    ref = continuous_raw.copy().filter(1.0, 40.0, method="fir", phase="zero").get_data()
    np.testing.assert_array_equal(ours, ref)

    ours_n = notch(continuous_raw, 50.0).get_data()
    ref_n = continuous_raw.copy().notch_filter(50.0, method="fir", phase="zero").get_data()
    np.testing.assert_array_equal(ours_n, ref_n)


# --- metadata, immutability, determinism -----------------------------------


@pytest.mark.parametrize("op", [lambda r: bandpass(r, 1.0, 40.0), lambda r: notch(r, 50.0)])
def test_metadata_preserved_and_input_untouched(continuous_raw, op):
    before = continuous_raw.get_data().copy()
    out = op(continuous_raw)
    assert out is not continuous_raw
    np.testing.assert_array_equal(continuous_raw.get_data(), before)
    assert out.get_data().shape == before.shape
    assert out.info["sfreq"] == continuous_raw.info["sfreq"]
    assert out.ch_names == continuous_raw.ch_names
    assert out.get_channel_types() == continuous_raw.get_channel_types()
    np.testing.assert_array_equal(op(continuous_raw).get_data(), out.get_data())


def test_bandpass_records_cutoffs_in_info(continuous_raw):
    out = bandpass(continuous_raw, 1.0, 40.0)
    assert (out.info["highpass"], out.info["lowpass"]) == (1.0, 40.0)
    assert continuous_raw.info["highpass"] == 0.0  # original untouched


def test_picks_limits_filtered_channels(continuous_raw):
    out = bandpass(continuous_raw, 1.0, 40.0, picks=["Fz"])
    np.testing.assert_array_equal(out.get_data()[1:], continuous_raw.get_data()[1:])
    assert not np.array_equal(out.get_data()[0], continuous_raw.get_data()[0])


# --- invalid inputs ---------------------------------------------------------


@pytest.mark.parametrize(
    "l_freq, h_freq",
    [(None, None), (40.0, 1.0), (10.0, 10.0), (0.0, 40.0), (-1.0, 40.0),
     (1.0, 125.0), (1.0, 200.0), (np.nan, 40.0), (1.0, np.inf)],
)
def test_bandpass_invalid_cutoffs(continuous_raw, l_freq, h_freq):
    with pytest.raises(ValueError):
        bandpass(continuous_raw, l_freq, h_freq)


@pytest.mark.parametrize("freqs", [[], 125.0, [50.0, 150.0], 0.0, -50.0, np.nan])
def test_notch_invalid_freqs(continuous_raw, freqs):
    with pytest.raises(ValueError):
        notch(continuous_raw, freqs)


def test_non_numeric_cutoff_type_error(continuous_raw):
    with pytest.raises(TypeError):
        bandpass(continuous_raw, "1", 40.0)


def test_rejects_epochs_and_arrays(continuous_raw):
    events = np.array([[500, 0, 1], [1500, 0, 1]])
    epochs = mne.Epochs(continuous_raw, events, tmin=0, tmax=1, baseline=None,
                        preload=True, verbose="error")
    with pytest.raises(TypeError, match="before epoching"):
        bandpass(epochs, 1.0, 40.0)
    with pytest.raises(TypeError, match="before epoching"):
        notch(epochs, 50.0)
    with pytest.raises(TypeError, match="to_raw"):
        bandpass(continuous_raw.get_data(), 1.0, 40.0)
