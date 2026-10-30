"""Resampling: rate and length bookkeeping, anti-aliasing, event timing, edge cases."""

import mne
import numpy as np
import pytest

from eeg_decoder.io import to_raw
from eeg_decoder.preprocessing import bandpass, resample

SFREQ = 250.0
N = 5000  # 20 s
EVENTS = [(250, 1), (1001, 2), (1751, 1), (3000, 3)]


def _raw(freqs=(10.0,), n=N, events=EVENTS):
    t = np.arange(n) / SFREQ
    sig = sum(10e-6 * np.sin(2 * np.pi * f * t) for f in freqs)
    stim = np.zeros(n)
    for s, code in events:
        stim[s] = code
    data = np.vstack([sig, 0.5 * sig, stim])
    return to_raw(data, SFREQ, ["Cz", "Pz", "STI"], ["eeg", "eeg", "stim"])


def _spectrum(raw):
    sf = raw.info["sfreq"]
    edge = int(2 * sf)
    x = raw.get_data()[0, edge:-edge]
    return np.fft.rfftfreq(x.size, 1 / sf), np.abs(np.fft.rfft(x)) * 2 / x.size


def _amp_at(raw, f):
    freqs, spec = _spectrum(raw)
    return spec[np.argmin(np.abs(freqs - f))]


# --- bookkeeping ------------------------------------------------------------


@pytest.mark.parametrize("new, n_new", [(100.0, 2000), (125.0, 2500), (500.0, 10000)])
def test_sfreq_and_length_updated(new, n_new):
    raw = _raw()
    out = resample(raw, new)
    assert out.info["sfreq"] == new
    assert out.n_times == n_new
    assert out.n_times / new == pytest.approx(raw.n_times / SFREQ)  # same duration


def test_non_integer_length_is_rounded():
    out = resample(_raw(n=2501, events=[]), 100.0)
    assert out.n_times == 1000  # round(2501 * 0.4) = round(1000.4)


def test_metadata_kept_and_input_untouched():
    raw = _raw()
    raw.info["bads"] = ["Pz"]
    before = raw.get_data().copy()
    out = resample(raw, 100.0)
    assert out is not raw
    np.testing.assert_array_equal(raw.get_data(), before)
    assert raw.info["sfreq"] == SFREQ
    assert out.ch_names == raw.ch_names
    assert out.get_channel_types() == raw.get_channel_types()
    assert out.info["bads"] == ["Pz"]


def test_lowpass_info_capped_at_new_nyquist():
    assert resample(_raw(), 100.0).info["lowpass"] == 50.0
    assert resample(bandpass(_raw(), 1.0, 30.0), 100.0).info["lowpass"] == 30.0


def test_same_rate_returns_equal_copy():
    raw = _raw()
    out = resample(raw, SFREQ)
    assert out is not raw
    np.testing.assert_array_equal(out.get_data(), raw.get_data())


@pytest.mark.parametrize("method", ["fft", "polyphase"])
def test_matches_mne_directly(method):
    raw = _raw()
    ref = raw.copy().resample(100.0, method=method, verbose="error").get_data()
    np.testing.assert_array_equal(resample(raw, 100.0, method=method).get_data(), ref)


# --- spectral content and anti-aliasing -------------------------------------


@pytest.mark.parametrize("method, min_gain", [("fft", 0.99), ("polyphase", 0.99)])
def test_passband_tone_preserved(method, min_gain):
    out = resample(_raw(freqs=(10.0,)), 100.0, method=method)
    assert _amp_at(out, 10.0) / 10e-6 == pytest.approx(1.0, abs=1 - min_gain)


@pytest.mark.parametrize("method", ["fft", "polyphase"])
def test_no_aliasing_above_new_nyquist(method):
    # 70 Hz is above the new Nyquist (50 Hz); without anti-aliasing it would
    # fold to 100 - 70 = 30 Hz. The 10 Hz tone is a reference that must survive.
    out = resample(_raw(freqs=(10.0, 70.0)), 100.0, method=method)
    assert _amp_at(out, 10.0) / 10e-6 == pytest.approx(1.0, abs=0.01)
    assert _amp_at(out, 30.0) / 10e-6 < 0.01  # alias suppressed by > 40 dB


def test_polyphase_rolls_off_near_new_nyquist():
    # Documented difference: polyphase attenuates content near the new Nyquist.
    fft = _amp_at(resample(_raw(freqs=(45.0,)), 100.0, method="fft"), 45.0) / 10e-6
    poly = _amp_at(resample(_raw(freqs=(45.0,)), 100.0, method="polyphase"), 45.0) / 10e-6
    assert fft > 0.98
    assert 0.85 < poly < 0.97


# --- event timing -----------------------------------------------------------


@pytest.mark.parametrize("new", [100.0, 125.0, 500.0])
def test_stim_events_remapped_within_half_a_sample(new):
    out = resample(_raw(), new)
    ev = mne.find_events(out, "STI", shortest_event=1, verbose="error")
    assert ev[:, 2].tolist() == [c for _, c in EVENTS]  # codes preserved, none lost
    expected_s = np.array([s for s, _ in EVENTS]) / SFREQ
    np.testing.assert_allclose(ev[:, 0] / new, expected_s, atol=0.5 / new + 1e-12)


def test_annotations_keep_onsets_in_seconds():
    raw = _raw()
    raw.set_annotations(mne.Annotations([1.004, 7.5], [0.5, 0.0], ["a", "b"]))
    out = resample(raw, 100.0)
    np.testing.assert_allclose(out.annotations.onset, [1.004, 7.5])


def test_merging_events_raise_instead_of_disappearing():
    raw = _raw(events=[(250, 1), (251, 2), (1000, 3)])  # 4 ms apart
    with pytest.raises(ValueError, match="merge events"):
        resample(raw, 100.0)  # 10 ms samples
    out = resample(raw, 500.0)  # upsampling keeps them apart
    assert mne.find_events(out, "STI", shortest_event=1, verbose="error")[:, 2].tolist() == [1, 2, 3]


def test_recording_without_stim_channel():
    raw = to_raw(np.zeros((2, 1000)), SFREQ, ["Fz", "Cz"])
    assert resample(raw, 100.0).n_times == 400


# --- invalid inputs ---------------------------------------------------------


@pytest.mark.parametrize("sfreq, exc", [(0, ValueError), (-100, ValueError),
                                        (np.nan, ValueError), (np.inf, ValueError),
                                        (True, TypeError), ("100", TypeError), (None, TypeError)])
def test_invalid_sfreq(sfreq, exc):
    with pytest.raises(exc):
        resample(_raw(), sfreq)


def test_invalid_method():
    with pytest.raises(ValueError, match="method"):
        resample(_raw(), 100.0, method="linear")


def test_empty_recording():
    info = mne.create_info(["Fz"], SFREQ, "eeg")
    with pytest.raises(ValueError, match="empty"):
        resample(mne.io.RawArray(np.zeros((1, 0)), info, verbose="error"), 100.0)


def test_rejects_epochs_and_arrays():
    raw = _raw()
    events = mne.find_events(raw, "STI", verbose="error")
    epochs = mne.Epochs(raw, events, tmin=0, tmax=0.5, baseline=None,
                        preload=True, verbose="error")
    with pytest.raises(TypeError, match="before epoching"):
        resample(epochs, 100.0)
    with pytest.raises(TypeError, match="to_raw"):
        resample(raw.get_data(), 100.0)
