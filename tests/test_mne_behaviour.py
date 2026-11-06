"""Pins of MNE-Python behaviours that EEGPrep-decoder works around.

Each test asserts what *MNE itself* does (not what our wrapper does), so a
change in a new MNE release fails exactly one test here and points at the
workaround to revisit. docs/mne-behaviours.md lists every behaviour, the
library's response, and the pinning test (some pins live next to the feature
they motivate, in other test files).
"""

import datetime

import mne
import numpy as np
import pytest

SF = 250.0


def _raw(n=2500, names=("a", "b", "c"), types="eeg", seed=0):
    rng = np.random.default_rng(seed)
    info = mne.create_info(list(names), SF, types)
    return mne.io.RawArray(rng.normal(0, 1e-5, (len(names), n)), info, verbose="error")


def _warnings(func):
    with mne.use_log_level("warning"), pytest.warns(RuntimeWarning) as record:
        result = func()
    return result, " ".join(str(w.message) for w in record)


# --- epoching -------------------------------------------------------------------------


def test_epochs_outside_the_data_are_dropped_silently():
    events = np.array([[100, 0, 1], [2450, 0, 1]])  # second window runs past the end
    ep = mne.Epochs(_raw(), events, {"x": 1}, -0.2, 0.5, baseline=None, preload=True,
                    verbose="error")
    assert len(ep) == 1 and ep.drop_log[1] in (("TOO_SHORT",), ("NO_DATA",))


def test_decimation_below_three_times_lowpass_only_warns():
    raw = _raw().filter(None, 40.0, verbose="error")
    _, msg = _warnings(lambda: mne.Epochs(raw, np.array([[500, 0, 1]]), {"x": 1}, -0.2, 0.5,
                                          baseline=None, decim=5))  # 50 Hz < 3 x 40 Hz
    assert "decim" in msg


def test_epochs_apply_a_baseline_by_default():
    ep = mne.Epochs(_raw(), np.array([[500, 0, 1]]), {"x": 1}, -0.2, 0.5, verbose="error")
    assert ep.baseline == (-0.2, 0.0)


def test_dropping_every_epoch_only_warns():
    ep = mne.Epochs(_raw(), np.array([[500, 0, 1]]), {"x": 1}, -0.2, 0.5, baseline=None,
                    preload=True, verbose="error")
    _, msg = _warnings(lambda: ep.drop_bad(reject={"eeg": 1e-9}))
    assert len(ep) == 0 and "All epochs were dropped" in msg


def test_epochs_metadata_without_pandas_is_a_plain_list_and_cannot_be_queried():
    try:
        import pandas  # noqa: F401
    except ImportError:
        ep = mne.Epochs(_raw(), np.array([[500, 0, 1], [1500, 0, 1]]), {"x": 1}, -0.2, 0.5,
                        preload=True, verbose="error")
        ep.metadata = [{"subject": "01"}, {"subject": "02"}]
        assert isinstance(ep.metadata, list)
        assert ep.copy().drop([0], verbose="error").metadata == [{"subject": "02"}]  # kept aligned
        with pytest.raises(KeyError):
            ep['subject == "01"']  # pandas-style selection needs pandas
    else:
        pytest.skip("pandas is installed; this pin needs an environment without it.")


# --- filtering, referencing, resampling ----------------------------------------------------


def test_fir_filter_longer_than_the_signal_only_warns():
    _, msg = _warnings(lambda: _raw(n=300).filter(1.0, 40.0))
    assert "longer than the signal" in msg


def test_filter_picks_none_covers_eeg_including_bads_but_not_other_types():
    raw = _raw(names=("Fz", "Pz", "EOG", "ECG", "STI"), types=["eeg", "eeg", "eog", "ecg", "stim"])
    raw.info["bads"] = ["Pz"]
    out = raw.copy().filter(1.0, 40.0, verbose="error")
    changed = [n for i, n in enumerate(raw.ch_names)
               if not np.array_equal(raw.get_data()[i], out.get_data()[i])]
    assert changed == ["Fz", "Pz"]


def test_average_reference_leaves_bad_channels_unreferenced():
    raw = _raw(names=("a", "b", "c", "d"))
    raw.info["bads"] = ["d"]
    out, _ = mne.set_eeg_reference(raw.copy(), "average", verbose="error")
    np.testing.assert_array_equal(out.get_data(picks="d", exclude=[]), raw.get_data(picks="d", exclude=[]))


def test_resampling_merges_stim_events_closer_than_one_new_sample():
    data = np.zeros((2, 10000))
    data[0] = np.random.default_rng(1).normal(0, 1e-5, 10000)
    data[1, [1000, 1003]] = [1, 2]  # 3 ms apart
    raw = mne.io.RawArray(data, mne.create_info(["a", "STI"], 1000.0, ["eeg", "stim"]),
                          verbose="error")
    out = raw.copy().resample(100.0, verbose="error")  # 10 ms samples
    assert mne.find_events(out, "STI", shortest_event=1, verbose="error")[:, 2].tolist() == [1]


# --- ICA --------------------------------------------------------------------------------------


def test_ica_apply_removes_the_union_with_ica_exclude():
    rng = np.random.default_rng(0)
    sources = np.vstack([np.sign(np.sin(np.arange(5000) / 7)), rng.laplace(size=(3, 5000))])
    raw = mne.io.RawArray(rng.normal(size=(4, 4)) @ sources * 1e-5,
                          mne.create_info(4, SF, "eeg"), verbose="error")
    ica = mne.preprocessing.ICA(n_components=4, method="infomax", random_state=0, max_iter=500,
                                verbose="error")
    ica.fit(raw, verbose="error")
    ica.exclude = [0]
    union = ica.apply(raw.copy(), exclude=[1], verbose="error").get_data()
    ica.exclude = []
    both = ica.apply(raw.copy(), exclude=[0, 1], verbose="error").get_data()
    np.testing.assert_allclose(union, both, rtol=0, atol=1e-20)


# --- saving and montages ------------------------------------------------------------------------


def test_plain_fif_save_keeps_measurement_date_and_subject_info(tmp_path):
    raw = _raw()
    raw.set_meas_date(datetime.datetime(2024, 5, 6, tzinfo=datetime.timezone.utc))
    raw.info["subject_info"] = {"first_name": "Jane"}
    ep = mne.Epochs(raw, np.array([[500, 0, 1]]), {"x": 1}, -0.2, 0.5, baseline=None,
                    preload=True, verbose="error")
    ep.save(tmp_path / "x-epo.fif", verbose="error")
    back = mne.read_epochs(tmp_path / "x-epo.fif", verbose="error")
    assert back.info["meas_date"].year == 2024 and back.info["subject_info"]["first_name"] == "Jane"


def test_standard_1020_montage_name_is_deprecated():
    # Docs and tests use "colin27_1020"; MNE 1.13 says "standard_1020" goes in 1.14.
    with pytest.warns(FutureWarning, match="colin27_1020"):
        mne.channels.make_standard_montage("standard_1020")
