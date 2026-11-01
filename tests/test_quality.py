"""Quality metrics and plots: exact values on known signals, no hidden plotting state."""

import io
import json

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest
from matplotlib.figure import Figure

from eeg_decoder.artifacts import reject_epochs
from eeg_decoder.epoching import make_epochs
from eeg_decoder.io import to_raw
from eeg_decoder.preprocessing import notch
from eeg_decoder.quality import (
    EpochsQuality,
    RecordingQuality,
    epochs_quality,
    plot_channel_quality,
    plot_psd_comparison,
    plot_rejection_summary,
    recording_quality,
)

SF, N = 250.0, 250 * 20
EEG = ["Fz", "Cz", "Pz", "Oz"]


def _raw(line_uv=0.0, with_stim=False):
    rng = np.random.default_rng(0)
    t = np.arange(N) / SF
    x = rng.normal(0, 10e-6, (4, N)) + line_uv * 1e-6 * np.sin(2 * np.pi * 50 * t)
    names, types = list(EEG), ["eeg"] * 4
    if with_stim:
        stim = np.zeros((1, N))
        stim[0, np.arange(250, N - 250, 250)] = 1
        x, names, types = np.vstack([x, stim]), names + ["STI"], types + ["stim"]
    return to_raw(x, SF, names, types)


# --- recording_quality ------------------------------------------------------


def test_basic_fields_and_bad_channels():
    raw = _raw()
    raw.info["bads"] = ["Oz"]
    q = recording_quality(raw)
    assert isinstance(q, RecordingQuality)
    assert (q.sfreq, q.duration_s, q.eeg_channels) == (SF, N / SF, tuple(EEG))
    assert q.bad_channels == ("Oz",) and q.bad_channel_fraction == 0.25
    assert (q.highpass, q.lowpass, q.custom_ref_applied) == (0.0, SF / 2, False)
    assert set(q.channel_metrics) == {"robust_sd", "ptp"}
    assert q.skipped == {"line_noise_db": "line_freq not given"}


def test_amplitude_metrics_match_definitions():
    raw = _raw()
    q = recording_quality(raw)
    x = raw.get_data()
    q75, q25 = np.percentile(x, [75, 25], axis=1)
    np.testing.assert_allclose([q.channel_metrics["robust_sd"][c] for c in EEG], 0.7413 * (q75 - q25))
    np.testing.assert_allclose([q.channel_metrics["ptp"][c] for c in EEG], np.ptp(x, axis=1))
    assert 9e-6 < q.channel_metrics["robust_sd"]["Cz"] < 11e-6  # 10 uV noise


def test_bad_spans_union_clipped_and_excluded_from_metrics():
    raw = _raw()
    raw._data[0, 2500:2510] += 1e-3  # huge artifact inside a BAD span
    with pytest.warns(RuntimeWarning, match="outside the data range"):  # MNE clips 19.5-21.5
        raw.set_annotations(mne.Annotations(
            [9.9, 10.0, 19.5, 5.0], [0.3, 0.5, 2.0, 0.0], ["BAD_a", "BAD_b", "BAD_a", "stim"]))
    q = recording_quality(raw)
    # union of 9.9-10.2 and 10.0-10.5 = 0.6 s; 19.5-21.5 clipped to 19.5-20.0 = 0.5 s
    assert q.bad_span_s == pytest.approx(1.1)
    assert q.bad_span_fraction == pytest.approx(1.1 / 20)
    assert q.annotation_s["BAD_a"] == pytest.approx(0.8)
    assert q.annotation_s["stim"] == 0.0  # zero-duration marker
    assert q.channel_metrics["ptp"]["Fz"] < 1e-4  # artifact excluded


def test_bad_spans_in_cropped_recording_with_meas_date():
    raw = _raw()
    raw.set_meas_date(1_600_000_000)
    raw.crop(tmin=2.0)
    raw.set_annotations(mne.Annotations([raw.first_time + 1.0], [0.5], ["BAD_x"],
                                        orig_time=raw.annotations.orig_time))
    q = recording_quality(raw)
    assert q.bad_span_s == pytest.approx(0.5) and q.annotation_s["BAD_x"] == pytest.approx(0.5)


def test_line_noise_metric():
    noisy = recording_quality(_raw(line_uv=20.0), line_freq=50)
    clean = recording_quality(_raw(), line_freq=50)
    notched = recording_quality(notch(_raw(line_uv=20.0), 50.0), line_freq=50)
    for c in EEG:
        assert noisy.channel_metrics["line_noise_db"][c] > 15
        assert abs(clean.channel_metrics["line_noise_db"][c]) < 3
        assert notched.channel_metrics["line_noise_db"][c] < 0  # notch leaves a dip
    assert noisy.line_freq == 50.0 and "line_noise_db" not in noisy.skipped


def test_line_noise_skipped_with_reason():
    near_nyquist = to_raw(np.random.default_rng(1).normal(0, 1e-5, (2, 2200)), 110.0, ["Cz", "Pz"])
    assert "exceeds Nyquist" in recording_quality(near_nyquist, line_freq=50).skipped["line_noise_db"]
    short = _raw().crop(tmax=1.0)
    assert "2 s of good data" in recording_quality(short, line_freq=50).skipped["line_noise_db"]


def test_all_data_bad():
    raw = _raw()
    raw.set_annotations(mne.Annotations([0.0], [N / SF], ["BAD_all"]))
    q = recording_quality(raw, line_freq=50)
    assert q.bad_span_fraction == pytest.approx(1.0)
    assert "no data outside BAD" in q.skipped["channel_metrics"]
    assert np.isnan(q.channel_metrics["robust_sd"]["Cz"])
    assert q.to_dict()["channel_metrics"]["robust_sd"]["Cz"] is None


def test_report_serialization_summary_and_immutability():
    raw = _raw(line_uv=20.0)
    raw.info["bads"] = ["Oz"]
    q = recording_quality(raw, line_freq=50)
    d = q.to_dict()
    assert json.loads(json.dumps(d, allow_nan=False)) == d
    text = q.summary()
    assert "bad channels: 1 (25%) ['Oz']" in text and "line noise at 50 Hz" in text
    with pytest.raises(TypeError):
        q.channel_metrics["ptp"]["Cz"] = 0.0
    with pytest.raises(AttributeError):
        q.sfreq = 1.0


def test_recording_quality_input_untouched_and_deterministic():
    raw = _raw(line_uv=5.0)
    before = raw.get_data().copy()
    a, b = recording_quality(raw, line_freq=50), recording_quality(raw, line_freq=50)
    np.testing.assert_array_equal(raw.get_data(), before)
    assert a.to_dict() == b.to_dict()


@pytest.mark.parametrize("line_freq, exc", [(0, ValueError), (125.0, ValueError), (-50, ValueError),
                                            (np.nan, ValueError), ("50", TypeError), (True, TypeError)])
def test_invalid_line_freq(line_freq, exc):
    with pytest.raises(exc):
        recording_quality(_raw(), line_freq=line_freq)


def test_recording_quality_invalid_inputs():
    raw = _raw()
    with pytest.raises(TypeError, match="continuous"):
        recording_quality(mne.make_fixed_length_epochs(raw, 1.0, preload=True, verbose="error"))
    with pytest.raises(ValueError, match="No EEG"):
        recording_quality(to_raw(np.zeros((1, 100)), SF, ["EOG"], ["eog"]))
    empty = mne.io.RawArray(np.zeros((1, 0)), mne.create_info(["Cz"], SF, "eeg"), verbose="error")
    with pytest.raises(ValueError, match="empty"):
        recording_quality(empty)


# --- epochs_quality ---------------------------------------------------------


def _epochs_with_drops():
    raw = _raw(with_stim=True)
    raw._data[0, 2500:2520] += 300e-6  # artifact in the epoch at 10 s
    raw.set_annotations(mne.Annotations([5.0], [0.2], ["BAD_blink"]))
    events = mne.find_events(raw, "STI", shortest_event=1, verbose="error")
    events[1::2, 2] = 2  # two conditions
    ep = make_epochs(raw, events, {"a": 1, "b": 2}, -0.2, 0.5)
    return reject_epochs(ep, reject={"eeg": 150e-6})


def test_epochs_quality_counts_reasons_and_conditions():
    eq = epochs_quality(_epochs_with_drops())
    assert isinstance(eq, EpochsQuality)
    assert eq.n_events == 18 and eq.n_dropped == 2 and eq.n_kept == 16
    assert eq.drop_fraction == pytest.approx(2 / 18)
    assert dict(eq.reasons) == {"BAD_blink": 1, "Fz": 1}
    assert sum(eq.kept_per_condition.values()) == 16 and set(eq.kept_per_condition) == {"a", "b"}


def test_ignored_events_are_not_counted():
    raw = _raw(with_stim=True)
    events = mne.find_events(raw, "STI", shortest_event=1, verbose="error")
    events[::3, 2] = 9  # code not in event_id -> IGNORED
    ep = mne.Epochs(raw, events, {"a": 1}, -0.2, 0.5, baseline=None, preload=True, verbose="error")
    eq = epochs_quality(ep)
    assert eq.n_events == len(ep) and eq.n_dropped == 0 and "IGNORED" not in eq.reasons


def test_epochs_quality_serialization_and_errors():
    eq = epochs_quality(_epochs_with_drops())
    d = eq.to_dict()
    assert json.loads(json.dumps(d)) == d
    assert "16 of 18 epochs kept" in eq.summary() and "dropped for Fz: 1" in eq.summary()
    with pytest.raises(TypeError):
        epochs_quality(_raw())


# --- plots ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_pyplot_leaks():
    plt.close("all")
    yield
    assert plt.get_fignums() == [], "plotting functions must not create pyplot figures"


def _renders(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getbuffer().nbytes > 1000


def test_psd_comparison_plots_the_mean_spectrum_of_good_channels():
    before = _raw(line_uv=20.0)
    after = notch(before, 50.0)
    fig = plot_psd_comparison(before, after, fmax=100)
    assert isinstance(fig, Figure) and _renders(fig)
    ax = fig.axes[0]
    lines = ax.get_lines()
    assert [ln.get_label() for ln in lines] == ["before", "after"]
    f_before, db_before = lines[0].get_data()
    i50 = np.argmin(np.abs(f_before - 50))
    assert db_before[i50] - np.median(db_before) > 15  # line-noise peak visible before
    f_after, db_after = lines[1].get_data()
    assert db_after[np.argmin(np.abs(f_after - 50))] < np.median(db_after)  # removed after
    assert f_before.max() <= 100 and f_before.min() > 0


def test_psd_comparison_ignores_bad_channels():
    raw = _raw()
    noisy = raw.copy()
    noisy._data[3] *= 100
    noisy.info["bads"] = ["Oz"]
    raw.info["bads"] = ["Oz"]
    y_a = plot_psd_comparison(raw, raw).axes[0].get_lines()[0].get_ydata()
    y_b = plot_psd_comparison(noisy, noisy).axes[0].get_lines()[0].get_ydata()
    np.testing.assert_allclose(y_a, y_b)


@pytest.mark.parametrize("kwargs, exc", [({"fmax": 0}, ValueError), ({"fmax": 200}, ValueError),
                                         ({"labels": ("a",)}, ValueError), ({"labels": "ab"}, ValueError),
                                         ({"window_s": 0}, ValueError), ({"window_s": 30.0}, ValueError)])
def test_psd_comparison_invalid_arguments(kwargs, exc):
    with pytest.raises(exc):
        plot_psd_comparison(_raw(), _raw(), **kwargs)


def test_psd_comparison_rejects_epochs():
    raw = _raw()
    with pytest.raises(TypeError):
        plot_psd_comparison(raw, mne.make_fixed_length_epochs(raw, 1.0, preload=True, verbose="error"))


def test_channel_quality_marks_bad_channels():
    raw = _raw()
    raw._data[2] *= 5
    raw.info["bads"] = ["Pz"]
    fig = plot_channel_quality(recording_quality(raw))
    assert _renders(fig)
    ax = fig.axes[0]
    bars = ax.patches
    assert len(bars) == len(EEG)
    hatched = [b for b in bars if b.get_hatch()]
    assert len(hatched) == 1 and hatched[0].get_height() > 40  # Pz, ~50 uV
    assert [t.get_text() for t in ax.get_xticklabels()] == EEG
    assert ax.get_title() == "Per-channel robust standard deviation"
    assert {t.get_text() for t in ax.get_legend().get_texts()} >= {"good", "bad (info['bads'])"}


def test_channel_quality_metric_choice_and_errors():
    q = recording_quality(_raw(line_uv=20.0), line_freq=50)
    assert plot_channel_quality(q, "line_noise_db").axes[0].get_ylabel().startswith("Line noise")
    with pytest.raises(ValueError, match="not computed"):
        plot_channel_quality(recording_quality(_raw()), "line_noise_db")
    with pytest.raises(ValueError, match="metric"):
        plot_channel_quality(q, "variance")
    with pytest.raises(TypeError):
        plot_channel_quality(q.to_dict())


def test_rejection_summary_bars_and_folding():
    eq = epochs_quality(_epochs_with_drops())
    fig = plot_rejection_summary(eq)
    assert _renders(fig)
    ax = fig.axes[0]
    assert sorted(t.get_text() for t in ax.get_yticklabels()) == ["BAD_blink", "Fz"]
    assert "16 of 18 epochs kept" in ax.get_title()
    many = EpochsQuality(30, 10, 20, 2 / 3, {f"ch{i}": 10 - i for i in range(8)}, {"a": 10})
    labels = [t.get_text() for t in plot_rejection_summary(many, max_reasons=3).axes[0].get_yticklabels()]
    assert labels[::-1] == ["ch0", "ch1", "other"]


def test_rejection_summary_without_drops_and_errors():
    raw = _raw(with_stim=True)
    events = mne.find_events(raw, "STI", shortest_event=1, verbose="error")
    eq = epochs_quality(make_epochs(raw, events, {"a": 1}, -0.2, 0.5))
    fig = plot_rejection_summary(eq)
    assert "No epochs dropped" in [t.get_text() for t in fig.axes[0].texts]
    with pytest.raises(ValueError):
        plot_rejection_summary(eq, max_reasons=0)
    with pytest.raises(TypeError):
        plot_rejection_summary(eq.to_dict())


def test_plotting_does_not_change_backend_or_inputs():
    backend = matplotlib.get_backend()
    raw = _raw()
    before = raw.get_data().copy()
    plot_psd_comparison(raw, raw)
    assert matplotlib.get_backend() == backend
    np.testing.assert_array_equal(raw.get_data(), before)
