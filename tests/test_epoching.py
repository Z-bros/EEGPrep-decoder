"""Epoch extraction and baseline correction: exact alignment, MNE parity, edge cases."""

import mne
import numpy as np
import pytest

from eeg_decoder.epoching import baseline_correct, make_epochs
from eeg_decoder.io import to_raw
from eeg_decoder.preprocessing import bandpass

SFREQ = 250.0
N = 5000  # 20 s
EVENT_SAMPLES = [500, 1250, 2000, 2750, 3500]
EVENT_CODES = [1, 2, 1, 2, 3]
EVENT_ID = {"left": 1, "right": 2}
STEP_V = 1e-9  # ramp: value encodes the sample index exactly


def _ramp_raw(first_samp_s=0.0):
    """Cz = sample index * 1 nV (alignment ruler); Pz = noise; EOG; stim."""
    rng = np.random.default_rng(3)
    idx = np.arange(N)
    data = np.vstack([
        idx * STEP_V,
        rng.normal(0, 10e-6, N),
        rng.normal(0, 20e-6, N),
        np.zeros(N),
    ])
    data[3, EVENT_SAMPLES] = EVENT_CODES
    raw = to_raw(data, SFREQ, ["Cz", "Pz", "EOG", "STI"], ["eeg", "eeg", "eog", "stim"])
    if first_samp_s:
        raw.crop(tmin=first_samp_s)  # first_samp becomes non-zero
    return raw


def _events(raw):
    return mne.find_events(raw, "STI", shortest_event=1, verbose="error")


# --- alignment and shape ----------------------------------------------------


@pytest.mark.parametrize("first_samp_s", [0.0, 1.0])
def test_epochs_are_aligned_to_events_exactly(first_samp_s):
    raw = _ramp_raw(first_samp_s)
    events = _events(raw)
    tmin, tmax = -0.2, 0.5
    ep = make_epochs(raw, events, EVENT_ID, tmin, tmax)
    cz = ep.get_data(picks="Cz")[:, 0, :]
    used = events[np.isin(events[:, 2], [1, 2])]
    assert raw.first_samp == int(round(first_samp_s * SFREQ))
    i0 = ep.time_as_index(0.0)[0]
    for row, (sample, _, _) in zip(cz, used):
        # Ramp value = original (pre-crop) sample index, which equals the absolute
        # event sample because cropping only shifts first_samp.
        assert row[i0] / STEP_V == pytest.approx(sample, abs=1e-6)  # t=0 is the event
        assert row[0] / STEP_V == pytest.approx(sample + round(tmin * SFREQ), abs=1e-6)


def test_shape_times_and_labels():
    raw = _ramp_raw()
    ep = make_epochs(raw, _events(raw), EVENT_ID, -0.2, 0.5)
    n_samp = int(round(0.7 * SFREQ)) + 1  # both endpoints included
    assert ep.get_data().shape == (4, 4, n_samp)  # code 3 not in event_id: ignored
    assert ep.times[0] == pytest.approx(-0.2) and ep.times[-1] == pytest.approx(0.5)
    assert ep.event_id == EVENT_ID
    assert ep.events[:, 2].tolist() == [1, 2, 1, 2]
    assert ep.baseline is None  # no implicit baseline
    assert ep.ch_names == raw.ch_names
    assert ep.info["sfreq"] == SFREQ


def test_matches_mne_directly():
    raw = _ramp_raw()
    events = _events(raw)
    ours = make_epochs(raw, events, EVENT_ID, -0.2, 0.5).get_data()
    ref = mne.Epochs(raw, events, EVENT_ID, -0.2, 0.5, baseline=None, preload=True,
                     proj=False, verbose="error").get_data()
    np.testing.assert_array_equal(ours, ref)


def test_input_untouched():
    raw = _ramp_raw()
    events = _events(raw)
    before, ev_before = raw.get_data().copy(), events.copy()
    make_epochs(raw, events, EVENT_ID, -0.2, 0.5)
    np.testing.assert_array_equal(raw.get_data(), before)
    np.testing.assert_array_equal(events, ev_before)


# --- decimation -------------------------------------------------------------


def test_decim_allowed_after_lowpass():
    raw = bandpass(_ramp_raw(), None, 20.0)  # info lowpass 20 Hz
    ep = make_epochs(raw, _events(raw), EVENT_ID, -0.2, 0.5, decim=2)  # 125 >= 60
    assert ep.info["sfreq"] == 125.0
    assert ep.get_data().shape[-1] == int(np.ceil((int(round(0.7 * SFREQ)) + 1) / 2))


@pytest.mark.parametrize("lowpass, decim", [(None, 2), (20.0, 5)])
def test_decim_aliasing_raises(lowpass, decim):
    raw = _ramp_raw() if lowpass is None else bandpass(_ramp_raw(), None, lowpass)
    with pytest.raises(ValueError, match="aliasing"):
        make_epochs(raw, _events(raw), EVENT_ID, -0.2, 0.5, decim=decim)


@pytest.mark.parametrize("decim", [0, -1, 1.5, True])
def test_invalid_decim(decim):
    raw = _ramp_raw()
    with pytest.raises(ValueError, match="decim"):
        make_epochs(raw, _events(raw), EVENT_ID, -0.2, 0.5, decim=decim)


# --- windows, events, annotations -------------------------------------------


def test_window_outside_recording_raises_not_silently_dropped():
    raw = _ramp_raw()
    events = np.array([[30, 0, 1], [1250, 0, 2], [4990, 0, 1]])
    with pytest.raises(ValueError, match=r"\[30, 4990\]"):
        make_epochs(raw, events, EVENT_ID, -0.2, 0.5)


def test_window_exactly_at_edges_is_kept():
    raw = _ramp_raw()
    events = np.array([[50, 0, 1], [N - 1 - 125, 0, 2]])  # tmin=-0.2 -> 50 samples
    ep = make_epochs(raw, events, EVENT_ID, -0.2, 0.5)
    assert len(ep) == 2


def test_bad_annotations_drop_epochs_and_are_logged():
    raw = _ramp_raw()
    raw.set_annotations(mne.Annotations([1250 / SFREQ - 0.1], [0.2], ["BAD_blink"]))
    ep = make_epochs(raw, _events(raw), EVENT_ID, -0.2, 0.5)
    assert len(ep) == 3
    assert ep.drop_log[1] == ("BAD_blink",)


def test_all_epochs_in_bad_annotations_raises():
    raw = _ramp_raw()
    raw.set_annotations(mne.Annotations([0.0], [N / SFREQ - 0.01], ["BAD_all"]))
    with pytest.raises(ValueError, match="BAD annotations"):
        make_epochs(raw, _events(raw), EVENT_ID, -0.2, 0.5)


@pytest.mark.parametrize(
    "events, exc, match",
    [(np.zeros((0, 3), int), ValueError, "empty"),
     (np.array([500, 0, 1]), ValueError, "shape"),
     (np.array([[500, 0, 1, 0]]), ValueError, "shape"),
     (np.array([[500.0, 0, 1]]), TypeError, "integer"),
     (np.array([[1250, 0, 2], [500, 0, 1]]), ValueError, "sorted"),
     (np.array([[500, 0, 1], [500, 0, 2]]), ValueError, "same sample"),
     (np.array([[500, 0, 1], [N + 10, 0, 2]]), ValueError, "outside the recording")],
)
def test_invalid_events(events, exc, match):
    with pytest.raises(exc, match=match):
        make_epochs(_ramp_raw(), events, EVENT_ID, -0.2, 0.5)


@pytest.mark.parametrize(
    "event_id, exc",
    [({"x": 9}, ValueError), ({}, TypeError), ([1, 2], TypeError),
     ({1: 1}, TypeError), ({"left": "1"}, TypeError)],
)
def test_invalid_event_id(event_id, exc):
    raw = _ramp_raw()
    with pytest.raises(exc):
        make_epochs(raw, _events(raw), event_id, -0.2, 0.5)


@pytest.mark.parametrize("tmin, tmax, exc", [(0.5, -0.2, ValueError), (0.1, 0.1, ValueError),
                                             (np.nan, 0.5, ValueError), ("0", 0.5, TypeError)])
def test_invalid_window(tmin, tmax, exc):
    raw = _ramp_raw()
    with pytest.raises(exc):
        make_epochs(raw, _events(raw), EVENT_ID, tmin, tmax)


def test_rejects_non_raw_and_empty():
    raw = _ramp_raw()
    ep = make_epochs(raw, _events(raw), EVENT_ID, -0.2, 0.5)
    with pytest.raises(TypeError, match="continuous"):
        make_epochs(ep, _events(raw), EVENT_ID, -0.2, 0.5)
    with pytest.raises(TypeError, match="to_raw"):
        make_epochs(raw.get_data(), _events(raw), EVENT_ID, -0.2, 0.5)
    info = mne.create_info(["Cz"], SFREQ, "eeg")
    with pytest.raises(ValueError, match="empty"):
        make_epochs(mne.io.RawArray(np.zeros((1, 0)), info, verbose="error"),
                    np.array([[0, 0, 1]]), {"a": 1}, 0.0, 0.1)


# --- baseline correction ----------------------------------------------------


def _offset_epochs():
    """Epochs where each epoch/channel has a distinct DC offset plus a response."""
    raw = _ramp_raw()
    ep = make_epochs(raw, _events(raw), EVENT_ID, -0.2, 0.5)
    rng = np.random.default_rng(5)
    data = ep.get_data()
    data[:, :3] += rng.normal(0, 50e-6, (len(ep), 3, 1))  # EEG + EOG offsets
    return mne.EpochsArray(data, ep.info, events=ep.events, event_id=ep.event_id,
                           tmin=ep.tmin, baseline=None, verbose="error")


def test_baseline_zero_mean_over_window_for_data_channels():
    ep = _offset_epochs()
    out = baseline_correct(ep, (None, 0.0))
    pre = out.times <= 0
    np.testing.assert_allclose(out.get_data()[:, :3][..., pre].mean(axis=-1), 0, atol=1e-18)
    np.testing.assert_array_equal(out.get_data()[:, 3], ep.get_data()[:, 3])  # stim unchanged
    assert out.baseline == pytest.approx((-0.2, 0.0))


def test_baseline_equals_manual_subtraction_and_mne():
    ep = _offset_epochs()
    x = ep.get_data()
    win = (ep.times >= -0.1) & (ep.times <= 0.0)
    manual = x[:, :3] - x[:, :3][..., win].mean(axis=-1, keepdims=True)
    ours = baseline_correct(ep, (-0.1, 0.0)).get_data()
    np.testing.assert_allclose(ours[:, :3], manual, atol=1e-18)
    np.testing.assert_array_equal(ours, ep.copy().apply_baseline((-0.1, 0.0), verbose="error").get_data())


def test_baseline_input_untouched():
    ep = _offset_epochs()
    before = ep.get_data().copy()
    out = baseline_correct(ep, (None, 0.0))
    assert out is not ep and ep.baseline is None
    np.testing.assert_array_equal(ep.get_data(), before)


@pytest.mark.parametrize(
    "baseline, exc",
    [((0.0, -0.1), ValueError), ((-1.0, 0.0), ValueError), ((0.0, 2.0), ValueError),
     ((0.0, 0.0), ValueError), ([None, 0.0], TypeError), ((None,), TypeError),
     (None, TypeError), (("a", 0.0), TypeError)],
)
def test_invalid_baseline(baseline, exc):
    with pytest.raises(exc):
        baseline_correct(_offset_epochs(), baseline)


def test_baseline_rejects_continuous_data():
    with pytest.raises(TypeError, match="high-pass"):
        baseline_correct(_ramp_raw(), (None, 0.0))
