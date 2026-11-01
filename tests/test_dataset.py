"""Dataset preparation: consistency checks, metadata alignment, .npz round trip."""

import importlib.util
import json

import mne
import numpy as np
import pytest

from eeg_decoder.dataset import (
    EpochDataset,
    check_compatible,
    combine_epochs,
    load_npz,
    match_channels,
    prepare_recordings,
    save_npz,
)
from eeg_decoder.epoching import make_epochs
from eeg_decoder.io import to_raw
from eeg_decoder.preprocessing import PreprocessingConfig, bandpass, rereference

SF, N = 250.0, 250 * 20
EEG = ["Fz", "Cz", "Pz", "Oz"]
EVENT_ID = {"left": 1, "right": 2}
CONFIG = PreprocessingConfig([
    {"op": "bandpass", "l_freq": 1.0, "h_freq": 40.0},
    {"op": "make_epochs", "event_source": "stim", "stim_channel": "STI",
     "event_id": EVENT_ID, "tmin": -0.2, "tmax": 0.5},
    {"op": "reject_epochs", "reject": {"eeg": 150e-6}},
])
N_EVENTS = len(np.arange(500, N - 500, 500))  # 8 events per recording


def _raw(seed=0, artifact=False, names=EEG, sfreq=SF):
    rng = np.random.default_rng(seed)
    n = int(N * sfreq / SF)
    x = rng.normal(0, 8e-6, (len(names), n))
    stim = np.zeros(n)
    idx = (np.arange(500, N - 500, 500) * sfreq / SF).astype(int)
    stim[idx] = np.resize([1, 2], idx.size)
    if artifact:
        x[0, idx[4] + 10 : idx[4] + 30] += 400e-6  # rejects exactly one epoch
    return to_raw(np.vstack([x, stim]), sfreq, list(names) + ["STI"], ["eeg"] * len(names) + ["stim"])


def _epochs(raw, event_id=EVENT_ID, tmin=-0.2, tmax=0.5):
    events = mne.find_events(raw, "STI", shortest_event=1, verbose="error")
    return make_epochs(raw, events, event_id, tmin, tmax)


def _recordings():
    return [
        (_raw(0), {"subject": "01", "session": 1}),
        (_raw(1, artifact=True), {"subject": "01", "session": 2}),
        (_raw(2), {"subject": "02", "session": 1}),
    ]


# --- check_compatible -------------------------------------------------------


def test_compatible_recordings_return_event_id_union():
    a = _epochs(_raw(0))
    b = _epochs(_raw(1), event_id={"left": 1})  # missing a condition is fine
    assert check_compatible([a, b]) == EVENT_ID


def test_all_incompatibilities_are_reported_together():
    ref = _epochs(_raw(0))
    other = _epochs(bandpass(_raw(1, sfreq=500.0), 1.0, 40.0), tmin=-0.1)
    other.info["bads"] = ["Cz"]
    with pytest.raises(ValueError) as info:
        check_compatible([ref, other], labels=["sub-01", "sub-02"])
    msg = str(info.value)
    for fragment in ("sampling rate", "epoch start tmin", "samples per epoch", "highpass",
                     "lowpass", "bad channels", "recording 1 (sub-02)"):
        assert fragment in msg
    assert msg.startswith("6 incompatibilities")  # all six planted differences, at once


@pytest.mark.parametrize("names, fragment", [(["Cz", "Fz", "Pz", "Oz"], "different order"),
                                             (["Fz", "Cz", "Pz", "O1"], "channel set differs")])
def test_channel_mismatches(names, fragment):
    with pytest.raises(ValueError, match=fragment):
        check_compatible([_epochs(_raw(0)), _epochs(_raw(1, names=names))])


def test_reference_baseline_and_event_code_conflicts():
    ref = _epochs(_raw(0))
    with pytest.raises(ValueError, match="custom reference"):
        check_compatible([ref, _epochs(rereference(_raw(1), "average"))])
    with pytest.raises(ValueError, match="baseline"):
        check_compatible([ref, ref.copy().apply_baseline((None, 0), verbose="error")])
    swapped = _epochs(_raw(1))
    swapped.event_id = {"left": 2, "right": 1}
    with pytest.raises(ValueError, match="'left' has code 2"):
        check_compatible([ref, swapped])


def test_empty_and_invalid_inputs():
    empty = _epochs(_raw(0))
    empty.drop(list(range(len(empty))), verbose="error")
    with pytest.raises(ValueError, match="no epochs"):
        check_compatible([_epochs(_raw(0)), empty])
    with pytest.raises(ValueError, match="empty"):
        check_compatible([])
    with pytest.raises(TypeError):
        check_compatible(_epochs(_raw(0)))
    with pytest.raises(TypeError):
        check_compatible([_raw(0)])


# --- match_channels ---------------------------------------------------------


def test_match_channels_reorders_and_selects_explicitly():
    ep = _epochs(_raw(1, names=["Cz", "Fz", "Pz", "Oz"]))
    out = match_channels(ep, ["Fz", "Cz", "Pz", "Oz"])
    assert out.ch_names == ["Fz", "Cz", "Pz", "Oz"]
    np.testing.assert_array_equal(out.get_data(picks="Fz"), ep.get_data(picks="Fz"))
    assert ep.ch_names[0] == "Cz"  # input untouched
    check_compatible([_epochs(_raw(0)).pick(["Fz", "Cz", "Pz", "Oz"]), out])
    assert match_channels(_raw(0), ["Oz", "Fz"]).ch_names == ["Oz", "Fz"]


@pytest.mark.parametrize("names, exc", [(["XX"], ValueError), ([], ValueError),
                                        (["Fz", "Fz"], ValueError), ("Fz", TypeError)])
def test_match_channels_invalid(names, exc):
    with pytest.raises(exc):
        match_channels(_raw(0), names)


# --- combine_epochs ---------------------------------------------------------


def test_combine_epochs_shapes_labels_and_identifiers():
    eps = [_epochs(_raw(i)) for i in range(3)]
    meta = [{"subject": "01"}, {"subject": "01"}, {"subject": "02"}]
    ds = combine_epochs(eps, metadata=meta, ch_types=["eeg"])
    assert isinstance(ds, EpochDataset)
    assert ds.X.shape == (3 * N_EVENTS, 4, 176) and ds.ch_names == tuple(EEG)
    np.testing.assert_array_equal(ds.X[:N_EVENTS], eps[0].get_data(picks="eeg"))
    np.testing.assert_array_equal(ds.y, np.concatenate([e.events[:, 2] for e in eps]))
    np.testing.assert_array_equal(ds.recording, np.repeat([0, 1, 2], N_EVENTS))
    assert ds.groups["subject"].tolist() == ["01"] * 2 * N_EVENTS + ["02"] * N_EVENTS
    assert ds.groups["subject"].dtype.kind == "U"
    assert set(ds.y_names) == {"left", "right"} and dict(ds.event_id) == EVENT_ID
    assert ds.records == [None, None, None]
    np.testing.assert_allclose(ds.times, eps[0].times)


def test_combine_keeps_all_channels_unless_asked():
    ds = combine_epochs([_epochs(_raw(0))])
    assert ds.ch_names == tuple(EEG) + ("STI",) and ds.ch_types[-1] == "stim"


def test_dataset_is_immutable():
    ds = combine_epochs([_epochs(_raw(0))], metadata=[{"subject": "01"}])
    for arr in (ds.X, ds.y, ds.recording, ds.times, ds.groups["subject"]):
        with pytest.raises(ValueError):
            arr[0] = arr[1]
    with pytest.raises(TypeError):
        ds.groups["subject"] = None
    with pytest.raises(AttributeError):
        ds.sfreq = 1.0


@pytest.mark.parametrize(
    "metadata, exc",
    [([{"subject": "01"}], ValueError), ([{"s": "01"}, {"t": "01"}], ValueError),
     ([{"s": [1]}, {"s": [2]}], TypeError), ([{"s": np.int64(1)}, {"s": 2}], TypeError),
     ([{"s": "01"}, {"s": 2}], ValueError), ([{"s": float("nan")}, {"s": 1.0}], ValueError),
     ([{1: "a"}, {1: "b"}], TypeError), ({"s": "01"}, ValueError)],
)
def test_combine_rejects_bad_metadata(metadata, exc):
    with pytest.raises(exc):
        combine_epochs([_epochs(_raw(0)), _epochs(_raw(1))], metadata=metadata)


def test_combine_rejects_bad_records_and_ch_types():
    eps = [_epochs(_raw(0))]
    with pytest.raises(ValueError, match="JSON"):
        combine_epochs(eps, records=[{"x": np.arange(3)}])
    with pytest.raises(TypeError):
        combine_epochs(eps, ch_types="eeg")
    with pytest.raises(ValueError, match="no channels of type"):
        combine_epochs(eps, ch_types=["eog"])


# --- prepare_recordings -----------------------------------------------------


def test_prepare_recordings_end_to_end_with_rejection():
    recs = _recordings()
    ds = prepare_recordings(recs, CONFIG, ch_types=["eeg"])
    assert ds.n_epochs == 3 * N_EVENTS - 1  # one epoch rejected in recording 1
    assert np.bincount(ds.recording).tolist() == [N_EVENTS, N_EVENTS - 1, N_EVENTS]
    for i, (_, meta) in enumerate(recs):  # identifiers stay aligned after rejection
        assert set(ds.groups["subject"][ds.recording == i]) == {meta["subject"]}
        assert set(ds.groups["session"][ds.recording == i]) == {meta["session"]}
    assert ds.groups["session"].dtype == np.int64
    records = ds.records
    assert [r["metadata"] for r in records] == [m for _, m in recs]
    assert records[1]["steps"][-1]["n_dropped"] == 1
    assert ds.failures == ()
    assert f"{3 * N_EVENTS - 1} epochs from 3 recording(s)" in ds.summary()


def test_prepare_matches_manual_pipeline_and_leaves_inputs_untouched():
    from eeg_decoder.preprocessing import EEGPipeline

    recs = _recordings()
    before = [r.get_data().copy() for r, _ in recs]
    ds = prepare_recordings(recs, CONFIG, ch_types=["eeg"])
    manual, _ = EEGPipeline(CONFIG).run(recs[2][0])
    np.testing.assert_array_equal(ds.X[ds.recording == 2], manual.get_data(picks="eeg"))
    for (raw, _), b in zip(recs, before):
        np.testing.assert_array_equal(raw.get_data(), b)


def test_first_failure_raises_with_a_note_by_default():
    recs = _recordings()
    recs[1] = (to_raw(np.zeros((4, N)), SF, EEG), {"subject": "01", "session": 2})  # no stim
    with pytest.raises(ValueError, match="not found") as info:
        prepare_recordings(recs, CONFIG)
    assert any("recording 1" in note for note in info.value.__notes__)


def test_collect_mode_skips_failures_without_shifting_indices():
    recs = _recordings()
    recs[1] = (to_raw(np.zeros((4, N)), SF, EEG), {"subject": "01", "session": 2})
    ds = prepare_recordings(recs, CONFIG, on_error="collect", ch_types=["eeg"])
    assert [f["index"] for f in ds.failures] == [1] and ds.failures[0]["error"] == "ValueError"
    assert sorted(set(ds.recording.tolist())) == [0, 2]
    assert ds.records[1] is None and ds.records[2]["metadata"] == {"subject": "02", "session": 1}
    assert len(ds.metadata) == 3
    assert set(ds.groups["subject"][ds.recording == 2]) == {"02"}


def test_collect_mode_treats_empty_result_as_failure_and_all_failed_raises():
    strict = PreprocessingConfig(list(CONFIG.to_dict()["steps"][:2]) + [
        {"op": "reject_epochs", "reject": {"eeg": 1e-7}}])
    with pytest.raises(ValueError, match="Every recording failed"):
        prepare_recordings(_recordings(), strict, on_error="collect")


def test_incompatible_results_always_raise():
    recs = _recordings()
    recs[2] = (_raw(2, names=["Fz", "Cz", "Pz", "O1"]), {"subject": "02", "session": 1})
    with pytest.raises(ValueError, match="channel set differs"):
        prepare_recordings(recs, CONFIG, on_error="collect")


@pytest.mark.parametrize(
    "args, kwargs, exc",
    [((_recordings(), PreprocessingConfig(CONFIG.to_dict()["steps"][:1])), {}, ValueError),
     ((_recordings(), CONFIG.to_dict()), {}, TypeError),
     ((_recordings(), CONFIG), {"on_error": "ignore"}, ValueError),
     (([], CONFIG), {}, TypeError), (([(_raw(0),)], CONFIG), {}, TypeError)],
)
def test_prepare_invalid_arguments(args, kwargs, exc):
    with pytest.raises(exc):
        prepare_recordings(*args, **kwargs)


# --- .npz export ------------------------------------------------------------


def _assert_same(a: EpochDataset, b: EpochDataset) -> None:
    for name in ("X", "y", "recording", "times"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        assert getattr(a, name).dtype == getattr(b, name).dtype
    assert dict(a.event_id) == dict(b.event_id)
    assert set(a.groups) == set(b.groups)
    for k in a.groups:
        np.testing.assert_array_equal(a.groups[k], b.groups[k])
        assert a.groups[k].dtype == b.groups[k].dtype
    assert (a.ch_names, a.ch_types, a.sfreq) == (b.ch_names, b.ch_types, b.sfreq)
    assert [dict(m) for m in a.metadata] == [dict(m) for m in b.metadata]
    assert a.records == b.records
    assert [dict(f) for f in a.failures] == [dict(f) for f in b.failures]


def test_npz_round_trip_is_exact_and_needs_no_pickle(tmp_path):
    recs = _recordings()
    recs[1] = (to_raw(np.zeros((4, N)), SF, EEG), {"subject": "01", "session": 2})
    ds = prepare_recordings(recs, CONFIG, on_error="collect", ch_types=["eeg"])
    path = save_npz(ds, tmp_path / "data.npz")
    back = load_npz(path)
    _assert_same(ds, back)
    with np.load(path, allow_pickle=False) as z:  # plain arrays only
        assert z["X"].shape == ds.X.shape
        assert json.loads(str(z["metadata_json"]))[0] == {"subject": "01", "session": 1}
    with pytest.raises(ValueError):
        back.X[0, 0, 0] = 1.0


def test_npz_overwrite_protection_and_path_checks(tmp_path):
    ds = combine_epochs([_epochs(_raw(0))])
    path = save_npz(ds, tmp_path / "d.npz")
    with pytest.raises(FileExistsError):
        save_npz(ds, path)
    save_npz(ds, path, overwrite=True)
    with pytest.raises(ValueError, match=".npz"):
        save_npz(ds, tmp_path / "d.npy")
    with pytest.raises(TypeError):
        save_npz(ds.X, tmp_path / "x.npz")
    with pytest.raises(FileNotFoundError):
        load_npz(tmp_path / "missing.npz")
    np.savez(tmp_path / "other.npz", X=np.zeros(3))
    with pytest.raises(ValueError, match="not an EpochDataset"):
        load_npz(tmp_path / "other.npz")


# --- optional pandas view ---------------------------------------------------


def test_to_dataframe_is_optional():
    ds = combine_epochs([_epochs(_raw(0))], metadata=[{"subject": "01"}])
    if importlib.util.find_spec("pandas") is None:
        with pytest.raises(ImportError, match="pandas"):
            ds.to_dataframe()
    else:
        df = ds.to_dataframe()
        assert list(df.columns) == ["recording", "condition", "code", "subject"]
        assert len(df) == ds.n_epochs


def test_dataset_subpackage_does_not_import_features_or_pandas():
    import subprocess
    import sys

    code = ("import sys, eeg_decoder.dataset; "
            "print('eeg_decoder.features' in sys.modules, 'pandas' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False False"
