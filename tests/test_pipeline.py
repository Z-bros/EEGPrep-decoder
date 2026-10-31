"""EEGPipeline and PreprocessingConfig: equivalence, provenance, validation."""

import json

import mne
import numpy as np
import pytest

from eeg_decoder.artifacts import reject_epochs
from eeg_decoder.epoching import baseline_correct, make_epochs
from eeg_decoder.io import to_raw
from eeg_decoder.preprocessing import (
    EEGPipeline,
    PreprocessingConfig,
    bandpass,
    notch,
    rereference,
    resample,
)

SFREQ = 250.0
N = 7500  # 30 s
EEG = ["Fz", "Cz", "Pz", "C3", "C4", "O1", "O2", "Oz"]
EVENT_SAMPLES = np.arange(750, 6751, 500)  # 13 events, 2 s apart
ARTIFACT_EVENT = 3  # index of the epoch with a planted 400 uV deflection

FULL_STEPS = [
    {"op": "notch", "freqs": [50.0, 100.0]},
    {"op": "bandpass", "l_freq": 1.0, "h_freq": 40.0},
    {"op": "rereference", "ref": "average"},
    {"op": "resample", "sfreq": 125.0},
    {"op": "make_epochs", "event_source": "stim", "stim_channel": "STI",
     "event_id": {"left": 1, "right": 2}, "tmin": -0.2, "tmax": 0.8},
    {"op": "baseline_correct", "baseline": [None, 0.0]},
    {"op": "reject_epochs", "reject": {"eeg": 150e-6}},
]


def _raw(with_stim=True):
    rng = np.random.default_rng(42)
    t = np.arange(N) / SFREQ
    eeg = rng.normal(0, 8e-6, (8, N)) + 15e-6 * np.sin(2 * np.pi * 10 * t)
    eeg += 5e-6 * np.sin(2 * np.pi * 50 * t)  # line noise
    s = EVENT_SAMPLES[ARTIFACT_EVENT]
    eeg[0, s + 10 : s + 30] += 400e-6
    names, types, data = list(EEG), ["eeg"] * 8, eeg
    if with_stim:
        stim = np.zeros((1, N))
        stim[0, EVENT_SAMPLES] = np.tile([1, 2], 7)[: EVENT_SAMPLES.size]
        names, types, data = names + ["STI"], types + ["stim"], np.vstack([eeg, stim])
    return to_raw(data, SFREQ, names, types)


def _manual_chain(raw):
    x = notch(raw, [50.0, 100.0])
    x = bandpass(x, 1.0, 40.0)
    x = rereference(x, "average")
    x = resample(x, 125.0)
    events = mne.find_events(x, "STI", shortest_event=1, consecutive=True, verbose="error")
    x = make_epochs(x, events, {"left": 1, "right": 2}, -0.2, 0.8)
    x = baseline_correct(x, (None, 0.0))
    return reject_epochs(x, reject={"eeg": 150e-6})


# --- equivalence with calling the functions by hand -------------------------


def test_full_pipeline_equals_manual_chain():
    raw = _raw()
    out, _ = EEGPipeline(PreprocessingConfig(FULL_STEPS)).run(raw)
    ref = _manual_chain(raw)
    assert isinstance(out, mne.BaseEpochs)
    np.testing.assert_array_equal(out.get_data(), ref.get_data())
    assert out.drop_log == ref.drop_log
    assert out.drop_log[ARTIFACT_EVENT] == ("Fz",)


def test_continuous_only_pipeline_returns_raw():
    raw = _raw()
    out, record = EEGPipeline(PreprocessingConfig(FULL_STEPS[:4])).run(raw)
    assert isinstance(out, mne.io.BaseRaw)
    ref = resample(rereference(bandpass(notch(raw, [50.0, 100.0]), 1.0, 40.0), "average"), 125.0)
    np.testing.assert_array_equal(out.get_data(), ref.get_data())
    assert record["steps"][-1]["output"]["type"] == "raw"


def test_epochs_input_with_epoch_only_steps():
    raw = _raw()
    events = mne.find_events(raw, "STI", shortest_event=1, verbose="error")
    epochs = make_epochs(raw, events, {"left": 1, "right": 2}, -0.2, 0.8)
    cfg = PreprocessingConfig([{"op": "rereference", "ref": ["Cz"]},
                               {"op": "baseline_correct", "baseline": [-0.2, 0.0]}])
    out, record = EEGPipeline(cfg).run(epochs)
    ref = baseline_correct(rereference(epochs, ["Cz"]), (-0.2, 0.0))
    np.testing.assert_array_equal(out.get_data(), ref.get_data())
    assert record["input"]["type"] == "epochs"


def test_annotation_event_source():
    raw = _raw(with_stim=False)
    onsets = EVENT_SAMPLES / SFREQ
    raw.set_annotations(mne.Annotations(onsets, 0.0, ["left", "right"] * 6 + ["left"]))
    cfg = PreprocessingConfig([
        {"op": "make_epochs", "event_source": "annotations",
         "event_id": {"left": 1, "right": 2}, "tmin": -0.2, "tmax": 0.8},
    ])
    out, record = EEGPipeline(cfg).run(raw)
    assert len(out) == EVENT_SAMPLES.size
    np.testing.assert_array_equal(out.events[:, 0], EVENT_SAMPLES)
    assert record["steps"][0]["n_events"] == EVENT_SAMPLES.size


def test_events_extracted_after_resampling():
    out, _ = EEGPipeline(PreprocessingConfig(FULL_STEPS[:5])).run(_raw())
    # 125 Hz: event samples halve; epoch length at the new rate
    np.testing.assert_array_equal(out.events[:, 0], EVENT_SAMPLES // 2)
    assert out.info["sfreq"] == 125.0
    assert len(out.times) == int(round(1.0 * 125.0)) + 1


# --- provenance record ------------------------------------------------------


def test_record_contents():
    raw = _raw()
    meta = {"dataset": "synthetic", "subject": "01", "session": 1}
    _, record = EEGPipeline(PreprocessingConfig(FULL_STEPS)).run(raw, metadata=meta)
    assert set(record) == {"schema_version", "config", "versions", "metadata", "input", "steps"}
    assert record["config"] == PreprocessingConfig(FULL_STEPS).to_dict()
    assert record["metadata"] == meta
    assert set(record["versions"]) == {"python", "eeg_decoder", "mne", "numpy", "scipy"}
    assert record["input"]["sfreq"] == SFREQ and record["input"]["n_times"] == N
    assert [s["op"] for s in record["steps"]] == [s["op"] for s in FULL_STEPS]
    by_op = {s["op"]: s for s in record["steps"]}
    assert by_op["notch"]["params"] == {"freqs": [50.0, 100.0]}  # notch now recorded
    assert by_op["rereference"]["params"] == {"ref": "average"}  # and which reference
    assert by_op["bandpass"]["output"]["lowpass"] == 40.0
    assert by_op["resample"]["output"]["sfreq"] == 125.0
    assert by_op["make_epochs"]["n_events"] == EVENT_SAMPLES.size
    assert by_op["make_epochs"]["n_dropped"] == 0
    assert by_op["reject_epochs"]["n_dropped"] == 1
    assert by_op["reject_epochs"]["output"]["n_epochs"] == EVENT_SAMPLES.size - 1
    assert by_op["baseline_correct"]["output"]["baseline"] == [-0.2, 0.0]


def test_record_is_json_serializable_and_deterministic():
    pipe = EEGPipeline(PreprocessingConfig(FULL_STEPS))
    out1, rec1 = pipe.run(_raw(), metadata={"subject": "01"})
    out2, rec2 = pipe.run(_raw(), metadata={"subject": "01"})
    assert json.loads(json.dumps(rec1, allow_nan=False)) == rec1
    assert rec1 == rec2
    np.testing.assert_array_equal(out1.get_data(), out2.get_data())


def test_record_excludes_identifying_info():
    raw = _raw()
    raw.set_meas_date(1_600_000_000)
    _, record = EEGPipeline(PreprocessingConfig(FULL_STEPS[:1])).run(raw)
    text = json.dumps(record)
    assert "meas_date" not in text and "subject_info" not in text


def test_input_untouched():
    raw = _raw()
    before = raw.get_data().copy()
    EEGPipeline(PreprocessingConfig(FULL_STEPS)).run(raw)
    np.testing.assert_array_equal(raw.get_data(), before)
    assert raw.info["sfreq"] == SFREQ and not raw.info["custom_ref_applied"]


# --- config: immutability and JSON ------------------------------------------


def test_config_roundtrip_dict_and_file(tmp_path):
    cfg = PreprocessingConfig(FULL_STEPS)
    assert PreprocessingConfig.from_dict(cfg.to_dict()) == cfg
    path = tmp_path / "preproc.json"
    cfg.save(path)
    loaded = PreprocessingConfig.load(path)
    assert loaded == cfg
    assert json.loads(path.read_text())["steps"] == FULL_STEPS
    out_a, _ = EEGPipeline(cfg).run(_raw())
    out_b, _ = EEGPipeline(loaded).run(_raw())
    np.testing.assert_array_equal(out_a.get_data(), out_b.get_data())


def test_config_is_immutable_and_detached_from_input():
    steps = [{"op": "bandpass", "l_freq": 1.0, "h_freq": 40.0}]
    cfg = PreprocessingConfig(steps)
    steps[0]["h_freq"] = 30.0  # mutating the caller's list does not change the config
    assert cfg.steps[0]["h_freq"] == 40.0
    with pytest.raises(TypeError):
        cfg.steps[0]["h_freq"] = 30.0
    with pytest.raises(AttributeError):
        cfg.steps = ()
    d = cfg.to_dict()
    d["steps"][0]["h_freq"] = 1.0  # to_dict returns an editable copy
    assert cfg.steps[0]["h_freq"] == 40.0
    assert cfg.ops == ("bandpass",)


def test_repr():
    assert repr(EEGPipeline(PreprocessingConfig(FULL_STEPS[:2]))) == "EEGPipeline(notch -> bandpass)"


# --- config validation ------------------------------------------------------


@pytest.mark.parametrize(
    "steps, exc, match",
    [
        ([], ValueError, "at least one"),
        ("bandpass", TypeError, "list"),
        ({"op": "bandpass"}, TypeError, "list"),
        ([["bandpass"]], TypeError, "mapping"),
        ([{"l_freq": 1.0}], ValueError, "no 'op'"),
        ([{"op": "highpass", "l_freq": 1.0}], ValueError, "unknown op"),
        ([{"op": "bandpass", "l_freq": 1.0, "hfreq": 40.0}], ValueError, "invalid parameters"),
        ([{"op": "bandpass", "l_freq": 1.0}], ValueError, "invalid parameters"),
        ([{"op": "bandpass", "l_freq": float("nan"), "h_freq": 40.0}], ValueError, "JSON"),
        ([{"op": "notch", "freqs": np.array([50.0])}], ValueError, "JSON"),
        ([{"op": "make_epochs", "event_source": "stim", "stim_channel": "STI",
           "events": [[1, 0, 1]], "event_id": {"a": 1}, "tmin": 0, "tmax": 1}],
         ValueError, "supplied by the pipeline"),
        ([{"op": "make_epochs", "event_source": "bids", "event_id": {"a": 1},
           "tmin": 0, "tmax": 1}], ValueError, "event_source"),
        ([{"op": "make_epochs", "event_source": "stim", "event_id": {"a": 1},
           "tmin": 0, "tmax": 1}], ValueError, "stim_channel"),
        ([{"op": "make_epochs", "event_source": "annotations", "stim_channel": "STI",
           "event_id": {"a": 1}, "tmin": 0, "tmax": 1}], ValueError, "only used"),
    ],
)
def test_invalid_config(steps, exc, match):
    with pytest.raises(exc, match=match):
        PreprocessingConfig(steps)


@pytest.mark.parametrize(
    "ops_steps, match",
    [
        ([FULL_STEPS[4], FULL_STEPS[1]], "runs on continuous"),  # filter after epoching
        ([FULL_STEPS[5], FULL_STEPS[4]], "runs on epoched"),  # baseline before epoching
        ([FULL_STEPS[4], FULL_STEPS[4]], "at most once"),
        ([FULL_STEPS[1], FULL_STEPS[5]], "runs on epoched"),  # baseline on raw
    ],
)
def test_invalid_order(ops_steps, match):
    with pytest.raises(ValueError, match=match):
        PreprocessingConfig(ops_steps)


@pytest.mark.parametrize(
    "data, exc",
    [({"steps": FULL_STEPS}, ValueError), ({"schema_version": 2, "steps": FULL_STEPS}, ValueError),
     ({"schema_version": 1, "steps": FULL_STEPS, "extra": 1}, ValueError), ([1], TypeError)],
)
def test_from_dict_rejects_bad_documents(data, exc):
    with pytest.raises(exc):
        PreprocessingConfig.from_dict(data)


# --- run-time errors --------------------------------------------------------


def test_input_type_must_match_config():
    raw = _raw()
    epoch_only = PreprocessingConfig([FULL_STEPS[5]])
    with pytest.raises(ValueError, match="cannot run on continuous"):
        EEGPipeline(epoch_only).run(raw)
    events = mne.find_events(raw, "STI", shortest_event=1, verbose="error")
    epochs = make_epochs(raw, events, {"left": 1, "right": 2}, -0.2, 0.8)
    with pytest.raises(ValueError, match="cannot run on epoched"):
        EEGPipeline(PreprocessingConfig(FULL_STEPS)).run(epochs)
    with pytest.raises(TypeError, match="to_raw"):
        EEGPipeline(epoch_only).run(raw.get_data())
    with pytest.raises(TypeError, match="PreprocessingConfig"):
        EEGPipeline(FULL_STEPS)


def test_step_errors_name_the_step():
    cfg = PreprocessingConfig([{"op": "notch", "freqs": [50.0]},
                               {"op": "bandpass", "l_freq": 1.0, "h_freq": 200.0}])
    with pytest.raises(ValueError, match="Nyquist") as info:
        EEGPipeline(cfg).run(_raw())
    assert any("step 1 (bandpass)" in note for note in info.value.__notes__)


@pytest.mark.parametrize("stim_channel, match", [("TRIG", "not found"), ("Cz", "not a stim")])
def test_bad_stim_channel(stim_channel, match):
    step = dict(FULL_STEPS[4], stim_channel=stim_channel)
    with pytest.raises(ValueError, match=match):
        EEGPipeline(PreprocessingConfig([step])).run(_raw())


def test_annotations_without_matches():
    raw = _raw()
    raw.set_annotations(mne.Annotations([1.0], [0.0], ["blink"]))
    cfg = PreprocessingConfig([{"op": "make_epochs", "event_source": "annotations",
                                "event_id": {"left": 1}, "tmin": -0.2, "tmax": 0.8}])
    with pytest.raises(ValueError, match=r"No annotations match.*\['blink'\]"):
        EEGPipeline(cfg).run(raw)
    with pytest.raises(ValueError, match=r"No annotations match.*\[\]"):
        EEGPipeline(cfg).run(_raw())  # no annotations at all: MNE returns empty instead


def test_stim_channel_without_events():
    raw = _raw()
    raw._data[-1] = 0.0  # empty stim channel
    with pytest.raises(ValueError, match="No events found on stim channel 'STI'"):
        EEGPipeline(PreprocessingConfig([FULL_STEPS[4]])).run(raw)


@pytest.mark.parametrize("metadata, exc", [({"x": np.float64(1.0)}, None), ({"x": {1, 2}}, ValueError),
                                           ({"x": float("nan")}, ValueError), (["s01"], TypeError)])
def test_metadata_validation(metadata, exc):
    pipe = EEGPipeline(PreprocessingConfig(FULL_STEPS[:1]))
    if exc is None:
        pipe.run(_raw(), metadata=metadata)  # numpy float subclasses float: fine
    else:
        with pytest.raises(exc):
            pipe.run(_raw(), metadata=metadata)
