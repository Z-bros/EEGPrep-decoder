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


# --- which channels picks=None touches (pins MNE 1.13 behaviour) ------------


def _mixed_raw():
    rng = np.random.default_rng(0)
    names = ["Fz", "Cz", "Pz", "EOG", "ECG", "EMG", "MISC", "STI"]
    types = ["eeg", "eeg", "eeg", "eog", "ecg", "emg", "misc", "stim"]
    data = rng.normal(0, 10e-6, (len(names), 5000))
    data[-1] = 0.0
    data[-1, ::500] = 1  # event codes
    raw = to_raw(data, SFREQ, names, types)
    raw.info["bads"] = ["Pz"]
    return raw


def _changed(before, after):
    return [n for i, n in enumerate(before.ch_names)
            if not np.array_equal(before.get_data()[i], after.get_data()[i])]


@pytest.mark.parametrize("op", [lambda r: bandpass(r, 1.0, 40.0), lambda r: notch(r, 50.0)])
def test_picks_none_filters_eeg_including_bads_only(op):
    raw = _mixed_raw()
    out = op(raw)
    assert _changed(raw, out) == ["Fz", "Cz", "Pz"]  # Pz is bad but still filtered
    assert out.info["bads"] == ["Pz"]
    np.testing.assert_array_equal(out.get_data(picks="STI"), raw.get_data(picks="STI"))


def test_explicit_picks_can_include_eog():
    raw = _mixed_raw()
    assert _changed(raw, bandpass(raw, 1.0, 40.0, picks=["eeg", "eog"])) == ["Fz", "Cz", "Pz", "EOG"]


def test_subset_picks_do_not_update_info_cutoffs(continuous_raw):
    out = bandpass(continuous_raw, 1.0, 40.0, picks=["Fz"])
    assert (out.info["highpass"], out.info["lowpass"]) == (0.0, SFREQ / 2)


# --- empty and short recordings ---------------------------------------------


def _short_raw(n_samples):
    t = np.arange(n_samples) / SFREQ
    return to_raw(np.tile(1e-5 * np.sin(2 * np.pi * 10 * t), (2, 1)), SFREQ, ["a", "b"])


@pytest.mark.parametrize("n_samples", [1, 5, 250, 824])
@pytest.mark.parametrize("op", [lambda r: bandpass(r, 1.0, 40.0), lambda r: notch(r, 50.0)])
def test_too_short_recording_raises(n_samples, op):
    # 1-40 Hz band-pass needs 825 samples, 50 Hz notch 1651 (at 250 Hz).
    raw = _short_raw(n_samples)
    before = raw.get_data().copy()
    with pytest.raises(ValueError, match="too short"):
        op(raw)
    np.testing.assert_array_equal(raw.get_data(), before)


def test_short_recording_ok_when_filter_fits():
    bandpass(_short_raw(825), 1.0, 40.0)  # exactly the filter length
    bandpass(_short_raw(250), None, 40.0)  # low-pass alone needs only 83 samples
    notch(_short_raw(1651), 50.0)


@pytest.mark.parametrize("level", ["error", "critical"])
def test_short_check_independent_of_mne_log_level(level):
    with mne.use_log_level(level):
        with pytest.raises(ValueError, match="too short"):
            bandpass(_short_raw(100), 1.0, 40.0)
        assert mne.utils.logger.getEffectiveLevel() >= 40  # caller's level restored


def test_empty_raw_rejected():
    raw = _short_raw(10).crop(tmax=0.0)  # MNE keeps one sample; build a 0-sample one
    empty = mne.io.RawArray(np.zeros((2, 0)), raw.info, verbose="error")
    with pytest.raises(ValueError, match="empty"):
        bandpass(empty, 1.0, 40.0)
    with pytest.raises(ValueError, match="empty"):
        notch(empty, 50.0)


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
