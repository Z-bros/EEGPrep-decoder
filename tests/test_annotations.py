"""Artifact annotations: detection of smooth blinks and flat stretches, time bases."""

import mne
import numpy as np
import pytest

from eeg_decoder.artifacts import add_annotations, annotate_artifacts
from eeg_decoder.epoching import make_epochs
from eeg_decoder.io import to_raw

SF, N = 250.0, 250 * 20
NAMES = ["Fp1", "Fz", "Cz", "Pz", "EOG", "STI"]
TYPES = ["eeg"] * 4 + ["eog", "stim"]
BLINK_T, FLAT_T = 6.0, 12.0  # s


def _raw():
    """5 uV noise; a smooth 300 uV Gaussian blink on Fp1 at 6 s; Pz flat 12-13 s."""
    rng = np.random.default_rng(0)
    t = np.arange(N) / SF
    x = rng.normal(0, 5e-6, (6, N))
    x[0] += 300e-6 * np.exp(-0.5 * ((t - BLINK_T) / 0.05) ** 2)
    x[3, int(FLAT_T * SF) : int((FLAT_T + 1) * SF)] = 0.0
    x[4] += 100e-6 * np.exp(-0.5 * ((t - BLINK_T) / 0.05) ** 2)
    x[5] = 0.0
    x[5, [int(s * SF) for s in (3, BLINK_T, 9, FLAT_T + 0.5, 16)]] = 1
    return to_raw(x, SF, NAMES, TYPES)


def _spans(ann, desc):
    """(onset, end) in seconds relative to the data start, for one description."""
    return [(o, o + d) for o, d, s in zip(ann.onset, ann.duration, ann.description) if s == desc]


# --- detection --------------------------------------------------------------


def test_smooth_blink_is_annotated_around_its_peak():
    ann = annotate_artifacts(_raw(), peak={"eeg": 150e-6}, window_s=0.5)
    spans = _spans(ann, "BAD_peak")
    assert len(spans) == 1
    start, end = spans[0]
    assert start < BLINK_T < end
    assert end - start < 1.0  # localized: blink plus about one window
    assert set(ann.description) == {"BAD_peak"}


def test_mne_annotate_amplitude_misses_the_same_blink():
    # Pins the MNE behaviour that motivates this function.
    mne_ann, _ = mne.preprocessing.annotate_amplitude(_raw(), peak={"eeg": 150e-6})
    assert not any(o < BLINK_T < o + d for o, d in zip(mne_ann.onset, mne_ann.duration))


def test_flat_stretch_is_annotated():
    ann = annotate_artifacts(_raw(), flat={"eeg": 1e-6}, window_s=0.2, min_duration_s=0.1)
    spans = _spans(ann, "BAD_flat")
    assert len(spans) == 1
    start, end = spans[0]
    assert FLAT_T - 0.01 <= start <= FLAT_T + 0.15 and FLAT_T + 0.85 <= end <= FLAT_T + 1.01


def test_peak_and_flat_together_and_clean_data():
    ann = annotate_artifacts(_raw(), peak={"eeg": 150e-6}, flat={"eeg": 1e-6}, window_s=0.2)
    assert sorted(set(ann.description)) == ["BAD_flat", "BAD_peak"]
    rng = np.random.default_rng(1)
    clean = to_raw(rng.normal(0, 5e-6, (3, N)), SF, ["Fz", "Cz", "Pz"])
    assert len(annotate_artifacts(clean, peak={"eeg": 150e-6}, flat={"eeg": 1e-6})) == 0


def test_thresholds_are_per_channel_type():
    raw = _raw()
    assert len(annotate_artifacts(raw, peak={"eog": 250e-6})) == 0  # EOG blink is 100 uV
    assert len(annotate_artifacts(raw, peak={"eog": 50e-6})) == 1


def test_bad_channels_are_ignored():
    raw = _raw()
    raw.info["bads"] = ["Fp1", "Pz"]
    assert len(annotate_artifacts(raw, peak={"eeg": 150e-6}, flat={"eeg": 1e-6})) == 0


def test_min_duration_and_padding():
    raw = _raw()
    base = _spans(annotate_artifacts(raw, peak={"eeg": 150e-6}, window_s=0.2), "BAD_peak")[0]
    padded = _spans(annotate_artifacts(raw, peak={"eeg": 150e-6}, window_s=0.2, pad_s=0.3),
                    "BAD_peak")[0]
    assert padded[0] == pytest.approx(base[0] - 0.3) and padded[1] == pytest.approx(base[1] + 0.3)
    assert len(annotate_artifacts(raw, flat={"eeg": 1e-6}, window_s=0.2, min_duration_s=2.0)) == 0


def test_padding_merges_overlapping_spans():
    rng = np.random.default_rng(2)
    t = np.arange(N) / SF
    x = rng.normal(0, 5e-6, (3, N))
    for c in (5.0, 5.8):  # two blinks 0.8 s apart
        x[0] += 300e-6 * np.exp(-0.5 * ((t - c) / 0.05) ** 2)
    raw = to_raw(x, SF, ["Fp1", "Fz", "Cz"])
    assert len(annotate_artifacts(raw, peak={"eeg": 150e-6}, window_s=0.2)) == 2
    assert len(annotate_artifacts(raw, peak={"eeg": 150e-6}, window_s=0.2, pad_s=0.5)) == 1


def test_input_untouched_and_deterministic():
    raw = _raw()
    before = raw.get_data().copy()
    a = annotate_artifacts(raw, peak={"eeg": 150e-6})
    b = annotate_artifacts(raw, peak={"eeg": 150e-6})
    np.testing.assert_array_equal(raw.get_data(), before)
    assert len(raw.annotations) == 0
    np.testing.assert_array_equal(a.onset, b.onset)


# --- time bases: annotations must drop exactly the overlapping epoch --------


@pytest.mark.parametrize("meas_date, crop_s", [(None, 0.0), (None, 2.0), (1_600_000_000, 2.0)])
def test_annotations_drop_the_right_epoch_in_every_time_base(meas_date, crop_s):
    raw = _raw()
    if meas_date is not None:
        raw.set_meas_date(meas_date)
    if crop_s:
        raw.crop(tmin=crop_s)  # first_samp becomes non-zero
    ann = annotate_artifacts(raw, peak={"eeg": 150e-6}, window_s=0.2)
    raw2 = add_annotations(raw, ann)
    events = mne.find_events(raw2, "STI", shortest_event=1, verbose="error")
    epochs = make_epochs(raw2, events, {"stim": 1}, tmin=-0.2, tmax=0.5)
    event_s = (events[:, 0] - raw.first_samp) / SF + crop_s
    dropped = [round(s, 3) for s, log in zip(event_s, epochs.drop_log) if log]
    assert dropped == [BLINK_T]


# --- add_annotations --------------------------------------------------------


def test_add_annotations_merges_and_copies():
    raw = _raw()
    raw.set_annotations(mne.Annotations([1.0], [0.5], ["note"]))
    new = annotate_artifacts(raw, peak={"eeg": 150e-6})
    out = add_annotations(raw, new)
    assert sorted(out.annotations.description) == ["BAD_peak", "note"]
    assert list(raw.annotations.description) == ["note"]


def test_add_annotations_rejects_mismatched_time_base():
    raw = _raw()
    other = _raw()
    other.set_meas_date(1_600_000_000)
    with pytest.raises(ValueError, match="time base"):
        add_annotations(raw, annotate_artifacts(other, peak={"eeg": 150e-6}))
    with pytest.raises(TypeError):
        add_annotations(raw, [(1.0, 0.5, "BAD")])


# --- invalid inputs ---------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, exc",
    [({}, ValueError), ({"peak": {"eeg": 150}}, ValueError), ({"peak": {"ecg": 1e-4}}, ValueError),
     ({"peak": {"eeg": 1e-4}, "window_s": 0}, ValueError),
     ({"peak": {"eeg": 1e-4}, "window_s": 0.004}, ValueError),
     ({"peak": {"eeg": 1e-4}, "window_s": 100.0}, ValueError),
     ({"peak": {"eeg": 1e-4}, "pad_s": -0.1}, ValueError),
     ({"peak": {"eeg": 1e-4}, "min_duration_s": np.nan}, ValueError),
     ({"peak": {"eeg": 1e-4}, "window_s": "0.5"}, TypeError)],
)
def test_invalid_arguments(kwargs, exc):
    with pytest.raises(exc):
        annotate_artifacts(_raw(), **kwargs)


def test_rejects_epochs_and_empty():
    raw = _raw()
    epochs = mne.make_fixed_length_epochs(raw, 1.0, preload=True, verbose="error")
    with pytest.raises(TypeError, match="continuous"):
        annotate_artifacts(epochs, peak={"eeg": 1e-4})
    empty = mne.io.RawArray(np.zeros((1, 0)), mne.create_info(["Cz"], SF, "eeg"), verbose="error")
    with pytest.raises(ValueError, match="empty"):
        annotate_artifacts(empty, peak={"eeg": 1e-4})
