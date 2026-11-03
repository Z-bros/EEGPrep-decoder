"""Per-recording .fif export: exact data, preserved MNE info, privacy, sidecars."""

import datetime
import json

import mne
import numpy as np
import pytest

from eeg_decoder.artifacts import reject_epochs
from eeg_decoder.dataset import (
    check_compatible,
    combine_epochs,
    load_epochs_fif,
    prepare_recordings,
    save_epochs_fif,
)
from eeg_decoder.epoching import baseline_correct, make_epochs
from eeg_decoder.io import to_raw
from eeg_decoder.preprocessing import PreprocessingConfig, bandpass, rereference

SF, N = 250.0, 250 * 40  # a 0.1 Hz high-pass needs > 33 s (8251 samples)
EEG = ["Fz", "Cz", "Pz", "Oz"]
CONFIG = PreprocessingConfig([
    {"op": "bandpass", "l_freq": 0.1, "h_freq": 40.0},  # 0.1 Hz is not exact in float32
    {"op": "rereference", "ref": "average"},
    {"op": "make_epochs", "event_source": "stim", "stim_channel": "STI",
     "event_id": {"left": 1, "right": 2}, "tmin": -0.2, "tmax": 0.5},
    {"op": "baseline_correct", "baseline": [None, 0.0]},
])


def _raw(seed=0, private=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 8e-6, (4, N))
    stim = np.zeros(N)
    idx = np.arange(500, N - 500, 500)
    stim[idx] = np.resize([1, 2], idx.size)
    raw = to_raw(np.vstack([x, stim]), SF, EEG + ["STI"], ["eeg"] * 4 + ["stim"])
    raw.set_montage("colin27_1020", on_missing="ignore")
    if private:
        raw.set_meas_date(datetime.datetime(2024, 5, 6, 7, 8, 9, tzinfo=datetime.timezone.utc))
        raw.info["subject_info"] = {"his_id": "patient-123", "first_name": "Jane"}
    return raw


def _epochs(seed=0, private=False):
    raw = rereference(bandpass(_raw(seed, private), 0.1, 40.0), "average")
    events = mne.find_events(raw, "STI", shortest_event=1, verbose="error")
    ep = baseline_correct(make_epochs(raw, events, {"left": 1, "right": 2}, -0.2, 0.5), (None, 0.0))
    return reject_epochs(ep, reject={"eeg": 1e-3})


# --- save_epochs_fif / load_epochs_fif --------------------------------------


def test_round_trip_is_exact_and_keeps_mne_information(tmp_path):
    ep = _epochs()
    fif, sidecar_path = save_epochs_fif(ep, tmp_path / "s01-epo.fif",
                                        metadata={"subject": "01"}, record={"steps": []})
    assert fif.name == "s01-epo.fif" and sidecar_path.name == "s01-epo.json"
    back, sidecar = load_epochs_fif(fif)
    np.testing.assert_array_equal(back.get_data(), ep.get_data())  # double precision
    np.testing.assert_array_equal(back.events, ep.events)
    assert back.event_id == ep.event_id and back.drop_log == ep.drop_log
    assert back.ch_names == ep.ch_names and back.get_montage() is not None
    assert back.info["custom_ref_applied"] and back.info["lowpass"] == 40.0
    assert sidecar["metadata"] == {"subject": "01"} and sidecar["record"] == {"steps": []}
    assert sidecar["anonymized"] is True and sidecar["fif"] == "s01-epo.fif"


def test_mne_default_single_precision_would_not_be_exact(tmp_path):
    # Pins the MNE behaviour this module avoids by saving with fmt="double".
    ep = _epochs()
    ep.save(tmp_path / "single-epo.fif", verbose="error")
    back = mne.read_epochs(tmp_path / "single-epo.fif", verbose="error")
    assert not np.array_equal(back.get_data(), ep.get_data())


def test_anonymized_by_default_and_opt_out(tmp_path):
    ep = _epochs(private=True)
    fif, _ = save_epochs_fif(ep, tmp_path / "a-epo.fif")
    back, _ = load_epochs_fif(fif)
    assert back.info["meas_date"].year == 2000
    assert back.info["subject_info"].get("first_name") != "Jane"
    assert back.info["subject_info"].get("his_id") != "patient-123"
    assert ep.info["subject_info"]["first_name"] == "Jane"  # input untouched
    fif2, _ = save_epochs_fif(ep, tmp_path / "b-epo.fif", anonymize=False)
    kept, sidecar = load_epochs_fif(fif2)
    assert kept.info["subject_info"]["first_name"] == "Jane" and sidecar["anonymized"] is False


def test_no_silent_overwrite(tmp_path):
    ep = _epochs()
    save_epochs_fif(ep, tmp_path / "x-epo.fif")
    with pytest.raises(FileExistsError):
        save_epochs_fif(ep, tmp_path / "x-epo.fif")
    (tmp_path / "y-epo.json").write_text("{}")  # a stray sidecar alone also blocks
    with pytest.raises(FileExistsError):
        save_epochs_fif(ep, tmp_path / "y-epo.fif")
    assert not (tmp_path / "y-epo.fif").exists()  # nothing written
    save_epochs_fif(ep, tmp_path / "x-epo.fif", overwrite=True)


@pytest.mark.parametrize(
    "path, kwargs, exc",
    [("bad.fif", {}, ValueError), ("bad-raw.fif", {}, ValueError), (3, {}, TypeError),
     ("ok-epo.fif", {"metadata": {"s": [1]}}, TypeError),
     ("ok-epo.fif", {"record": {"x": np.arange(2)}}, ValueError),
     ("ok-epo.fif", {"anonymize": "yes"}, TypeError)],
)
def test_save_invalid_arguments(tmp_path, path, kwargs, exc):
    target = tmp_path / path if isinstance(path, str) else path
    with pytest.raises(exc):
        save_epochs_fif(_epochs(), target, **kwargs)


def test_save_rejects_non_epochs_and_empty(tmp_path):
    with pytest.raises(TypeError):
        save_epochs_fif(_raw(), tmp_path / "r-epo.fif")
    empty = _epochs()
    empty.drop(list(range(len(empty))), verbose="error")
    with pytest.raises(ValueError, match="no epochs"):
        save_epochs_fif(empty, tmp_path / "e-epo.fif")


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_epochs_fif(tmp_path / "missing-epo.fif")
    _epochs().save(tmp_path / "plain-epo.fif", fmt="double", verbose="error")
    with pytest.raises(FileNotFoundError, match="Sidecar"):
        load_epochs_fif(tmp_path / "plain-epo.fif")
    (tmp_path / "plain-epo.json").write_text(json.dumps({"format": "other"}))
    with pytest.raises(ValueError, match="not an EEGPrep-decoder"):
        load_epochs_fif(tmp_path / "plain-epo.fif")
    with pytest.raises(ValueError, match="-epo.fif"):
        load_epochs_fif(tmp_path / "plain.fif")


# --- float32 header values must not break the consistency check -------------


def test_reloaded_epochs_are_still_compatible_with_fresh_ones(tmp_path):
    fresh = _epochs(0)
    fif, _ = save_epochs_fif(_epochs(1), tmp_path / "r-epo.fif")
    reloaded, _ = load_epochs_fif(fif)
    assert reloaded.baseline != fresh.baseline  # float32 in the file (MNE behaviour)
    assert reloaded.info["highpass"] != fresh.info["highpass"]  # 0.1 Hz as float32
    check_compatible([fresh, reloaded])  # tolerant comparison: no false mismatch


def test_rebuild_dataset_from_fif_files(tmp_path):
    paths = [save_epochs_fif(_epochs(i), tmp_path / f"s{i}-epo.fif",
                             metadata={"subject": f"0{i}"}, record={"i": i})[0] for i in range(2)]
    loaded = [load_epochs_fif(p) for p in paths]
    ds = combine_epochs([e for e, _ in loaded], metadata=[s["metadata"] for _, s in loaded],
                        records=[s["record"] for _, s in loaded], ch_types=["eeg"])
    direct = combine_epochs([_epochs(0), _epochs(1)], metadata=[{"subject": "00"}, {"subject": "01"}],
                            ch_types=["eeg"])
    np.testing.assert_array_equal(ds.X, direct.X)
    assert ds.records == [{"i": 0}, {"i": 1}]


# --- prepare_recordings(fif_dir=...) ----------------------------------------


def _recordings():
    return [(_raw(0, private=True), {"subject": "01"}), (_raw(1), {"subject": "02"})]


def test_prepare_writes_one_anonymized_file_per_recording(tmp_path):
    ds = prepare_recordings(_recordings(), CONFIG, ch_types=["eeg"], fif_dir=tmp_path)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["recording-000-epo.fif", "recording-000-epo.json",
                     "recording-001-epo.fif", "recording-001-epo.json"]
    ep0, side0 = load_epochs_fif(tmp_path / "recording-000-epo.fif")
    assert side0["metadata"] == {"subject": "01"} and side0["record"] == ds.records[0]
    assert ep0.info["meas_date"].year == 2000  # anonymized
    assert "STI" in ep0.ch_names  # full epochs, independent of ch_types
    np.testing.assert_array_equal(ep0.get_data(picks="eeg"), ds.X[ds.recording == 0])


def test_prepare_refuses_existing_targets_before_running(tmp_path):
    (tmp_path / "recording-001-epo.json").write_text("{}")
    with pytest.raises(FileExistsError, match="recording-001"):
        prepare_recordings(_recordings(), CONFIG, fif_dir=tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["recording-001-epo.json"]
    with pytest.raises(FileNotFoundError):
        prepare_recordings(_recordings(), CONFIG, fif_dir=tmp_path / "nope")
    with pytest.raises(TypeError):
        prepare_recordings(_recordings(), CONFIG, fif_dir=1)


def test_prepare_collect_mode_writes_only_successes(tmp_path):
    recs = _recordings() + [(to_raw(np.zeros((4, N)), SF, EEG), {"subject": "03"})]
    ds = prepare_recordings(recs, CONFIG, on_error="collect", fif_dir=tmp_path)
    assert [f["index"] for f in ds.failures] == [2]
    assert not (tmp_path / "recording-002-epo.fif").exists()
    assert (tmp_path / "recording-001-epo.fif").exists()


def test_nothing_written_when_recordings_are_incompatible(tmp_path):
    other = to_raw(np.vstack([np.random.default_rng(3).normal(0, 8e-6, (4, N)), _raw(0).get_data()[-1:]]),
                   SF, ["Fz", "Cz", "Pz", "O1", "STI"], ["eeg"] * 4 + ["stim"])
    with pytest.raises(ValueError, match="channel set differs"):
        prepare_recordings([_recordings()[0], (other, {"subject": "02"})], CONFIG, fif_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []
