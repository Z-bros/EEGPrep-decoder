"""Optional ICA: blink/heartbeat separation on synthetic mixtures with known sources."""

import importlib.util
import json

import mne
import numpy as np
import pytest

from eeg_decoder.artifacts import (
    ComponentReport,
    apply_ica,
    find_artifact_components,
    fit_ica,
)
from eeg_decoder.io import to_raw

SF, N = 250.0, 250 * 40
EEG = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "Cz", "P3", "P4", "Pz", "O1", "O2"]
N_COMP = 10  # 8 neural sources + blink + heartbeat


def _make():
    """12 EEG channels = random mix of 8 neural sources + frontal blink + heartbeat."""
    rng = np.random.default_rng(0)
    t = np.arange(N) / SF
    neural = np.vstack(
        [np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f in (6, 10, 12, 20)]
        + [rng.laplace(size=(4, N))]
    )
    blink = np.zeros(N)
    for c in np.arange(2, 40, 3.1) + rng.uniform(-0.3, 0.3, 13):
        blink += np.exp(-0.5 * ((t - c) / 0.08) ** 2)
    heart = np.zeros(N)
    for c in np.arange(0.4, 40, 0.85):
        heart += np.exp(-0.5 * ((t - c) / 0.01) ** 2)
    mix = rng.normal(0, 1, (12, 8)) * 10e-6
    w_blink = np.array([1, 1, .7, .7, .3, .3, .3, .1, .1, .1, .05, .05]) * 150e-6
    w_heart = rng.normal(0, 1, 12) * 4e-6
    eeg = (mix @ neural + np.outer(w_blink, blink) + np.outer(w_heart, heart)
           + rng.normal(0, 0.5e-6, (12, N)))
    eog = 200e-6 * blink + rng.normal(0, 2e-6, N)
    ecg = 500e-6 * heart + rng.normal(0, 5e-6, N)
    raw = to_raw(np.vstack([eeg, eog, ecg]), SF, EEG + ["EOG", "ECG"],
                 ["eeg"] * 12 + ["eog", "ecg"])
    return raw, {"neural": mix @ neural, "blink": blink, "heart": heart}


@pytest.fixture(scope="module")
def data():
    return _make()


@pytest.fixture(scope="module")
def ica(data):
    return fit_ica(data[0], n_components=N_COMP, seed=42)


def _corr(a, b):
    return abs(np.corrcoef(a, b)[0, 1])


# --- fitting ----------------------------------------------------------------


def test_fit_separates_artifact_sources(data, ica):
    raw, truth = data
    sources = ica.get_sources(raw).get_data()
    assert ica.n_components_ == N_COMP
    assert max(_corr(s, truth["blink"]) for s in sources) > 0.99
    assert max(_corr(s, truth["heart"]) for s in sources) > 0.9


def test_fit_on_highpassed_copy_and_input_untouched(data, ica):
    raw, _ = data
    assert ica.info["highpass"] == 1.0  # fitted on the 1 Hz copy
    assert raw.info["highpass"] == 0.0  # original untouched
    assert ica.exclude == [] and ica.ch_names == EEG  # EEG only; EOG/ECG not fitted


def test_same_seed_is_reproducible(data, ica):
    again = fit_ica(data[0], n_components=N_COMP, seed=42)
    np.testing.assert_array_equal(again.unmixing_matrix_, ica.unmixing_matrix_)


def test_bad_channels_excluded_from_fit_and_untouched_by_apply(data):
    raw, _ = data
    raw = raw.copy()
    raw.info["bads"] = ["O2"]
    ica = fit_ica(raw, n_components=N_COMP, seed=1)
    assert "O2" not in ica.ch_names
    clean = apply_ica(raw, ica, exclude=[0])
    np.testing.assert_array_equal(clean.get_data(picks="O2"), raw.get_data(picks="O2"))


def test_highpass_none_and_already_highpassed(data):
    from eeg_decoder.preprocessing import bandpass

    raw, _ = data
    assert fit_ica(raw, n_components=N_COMP, seed=0, highpass_hz=None).info["highpass"] == 0.0
    hp2 = bandpass(raw, 2.0, None, picks="eeg")
    assert fit_ica(hp2, n_components=N_COMP, seed=0).info["highpass"] == 2.0  # not refiltered


@pytest.mark.parametrize(
    "kwargs, exc",
    [({"n_components": 1, "seed": 0}, ValueError), ({"n_components": 13, "seed": 0}, ValueError),
     ({"n_components": 1.0, "seed": 0}, ValueError), ({"n_components": "10", "seed": 0}, TypeError),
     ({"n_components": 5, "seed": -1}, ValueError), ({"n_components": 5, "seed": 1.5}, ValueError),
     ({"n_components": 5, "seed": 0, "method": "jade"}, ValueError),
     ({"n_components": 5, "seed": 0, "highpass_hz": 0}, ValueError),
     ({"n_components": 5, "seed": 0, "highpass_hz": "1"}, TypeError)],
)
def test_fit_invalid_arguments(data, kwargs, exc):
    with pytest.raises(exc):
        fit_ica(data[0], **kwargs)


def test_seed_and_n_components_are_required(data):
    with pytest.raises(TypeError):
        fit_ica(data[0], n_components=5)
    with pytest.raises(TypeError):
        fit_ica(data[0], seed=0)


@pytest.mark.parametrize("method, module", [("picard", "picard"), ("fastica", "sklearn")])
def test_optional_methods_need_their_package(data, method, module):
    if importlib.util.find_spec(module) is None:
        with pytest.raises(ImportError, match=r"eeg-decoder\[ica\]"):
            fit_ica(data[0], n_components=5, seed=0, method=method)
    else:
        assert fit_ica(data[0], n_components=5, seed=0, method=method).n_components_ == 5


def test_fit_rejects_epochs_empty_and_too_few_channels(data):
    raw, _ = data
    epochs = mne.make_fixed_length_epochs(raw, 2.0, preload=True, verbose="error")
    with pytest.raises(TypeError, match="continuous"):
        fit_ica(epochs, n_components=5, seed=0)
    with pytest.raises(ValueError, match="at least 2"):
        fit_ica(to_raw(np.zeros((1, 1000)), SF, ["Cz"]), n_components=2, seed=0)
    empty = mne.io.RawArray(np.zeros((2, 0)), mne.create_info(["Cz", "Pz"], SF, "eeg"),
                            verbose="error")
    with pytest.raises(ValueError, match="empty"):
        fit_ica(empty, n_components=2, seed=0)


# --- finding components -----------------------------------------------------


def _blink_and_heart_components(raw, ica, truth):
    s = ica.get_sources(raw).get_data()
    return (int(np.argmax([_corr(x, truth["blink"]) for x in s])),
            int(np.argmax([_corr(x, truth["heart"]) for x in s])))


def test_correlation_measure_finds_blink_and_heart(data, ica):
    raw, truth = data
    blink_ic, heart_ic = _blink_and_heart_components(raw, ica, truth)
    rep = find_artifact_components(ica, raw, eog=["EOG"], ecg="ECG",
                                   measure="correlation", threshold=0.7)
    assert rep.eog == (blink_ic,)
    assert heart_ic in rep.ecg
    assert set(rep.exclude) == set(rep.eog) | set(rep.ecg)
    assert len(rep.scores["EOG"]) == N_COMP and rep.scores["EOG"][blink_ic] > 0.99


def test_zscore_measure_with_reachable_threshold(data, ica):
    raw, truth = data
    blink_ic, _ = _blink_and_heart_components(raw, ica, truth)
    rep = find_artifact_components(ica, raw, eog=["EOG"], threshold=2.5)
    assert blink_ic in rep.eog and rep.measure == "zscore"
    # MNE iterates the z-scoring, so a low threshold can add near-zero-correlation
    # components (documented); the scores make that visible for review.
    extra = [i for i in rep.eog if i != blink_ic]
    assert all(abs(rep.scores["EOG"][i]) < 0.2 for i in extra)


def test_unreachable_zscore_threshold_raises_instead_of_finding_nothing(data, ica):
    raw, _ = data
    # 10 components: max reachable z = sqrt(9) = 3, and MNE needs strictly more.
    with pytest.raises(ValueError, match=r"sqrt\(10 - 1\) = 3.00"):
        find_artifact_components(ica, raw, eog=["EOG"])
    # Pin the MNE behaviour this guards against: it silently returns nothing.
    idx, _ = ica.copy().find_bads_eog(raw, ch_name="EOG", threshold=3.0, verbose="error")
    assert list(idx) == []


def test_report_json_and_ica_untouched(data, ica):
    raw, _ = data
    rep = find_artifact_components(ica, raw, eog=["EOG", "Fp1"], measure="correlation",
                                   threshold=0.7)
    assert isinstance(rep, ComponentReport)
    d = rep.to_dict()
    assert json.loads(json.dumps(d, allow_nan=False)) == d
    assert set(d["scores"]) == {"EOG", "Fp1"}
    assert ica.exclude == [] and not getattr(ica, "labels_", {})


@pytest.mark.parametrize(
    "kwargs, exc",
    [({}, ValueError), ({"eog": "EOG"}, TypeError), ({"eog": []}, TypeError),
     ({"eog": ["XX"]}, ValueError), ({"ecg": ["ECG"]}, TypeError),
     ({"eog": ["EOG"], "measure": "pearson"}, ValueError),
     ({"eog": ["EOG"], "measure": "correlation", "threshold": 1.2}, ValueError),
     ({"eog": ["EOG"], "threshold": 0}, ValueError),
     ({"eog": ["EOG"], "threshold": "3"}, TypeError)],
)
def test_find_invalid_arguments(data, ica, kwargs, exc):
    with pytest.raises(exc):
        find_artifact_components(ica, data[0], **kwargs)


def test_unfitted_ica_rejected(data):
    with pytest.raises(ValueError, match="not fitted"):
        find_artifact_components(mne.preprocessing.ICA(n_components=2), data[0], eog=["EOG"])


# --- applying ---------------------------------------------------------------


def test_apply_removes_blink_and_keeps_neural_signal(data, ica):
    raw, truth = data
    blink_ic, heart_ic = _blink_and_heart_components(raw, ica, truth)
    clean = apply_ica(raw, ica, exclude=[blink_ic, heart_ic])
    fp1_before = raw.get_data(picks="Fp1")[0]
    fp1_after = clean.get_data(picks="Fp1")[0]
    assert _corr(fp1_before, truth["blink"]) > 0.6
    assert _corr(fp1_after, truth["blink"]) < 0.05
    assert _corr(fp1_after, truth["neural"][0]) > 0.99
    eeg_after = clean.get_data(picks="eeg")
    assert max(_corr(ch, truth["heart"]) for ch in eeg_after) < 0.05
    np.testing.assert_array_equal(clean.get_data(picks=["EOG", "ECG"]),
                                  raw.get_data(picks=["EOG", "ECG"]))


def test_apply_removes_exactly_the_listed_components(data, ica):
    raw, _ = data
    stale = ica.copy()
    stale.exclude = [0, 1]  # must be ignored
    ours = apply_ica(raw, stale, exclude=[2]).get_data()
    ref = ica.copy().apply(raw.copy(), exclude=[2], verbose="error").get_data()
    np.testing.assert_array_equal(ours, ref)
    assert stale.exclude == [0, 1]


def test_apply_empty_exclude_returns_unchanged_copy(data, ica):
    raw, _ = data
    out = apply_ica(raw, ica, exclude=[])
    assert out is not raw
    np.testing.assert_array_equal(out.get_data(), raw.get_data())


def test_apply_to_epochs_and_input_untouched(data, ica):
    raw, _ = data
    before = raw.get_data().copy()
    epochs = mne.make_fixed_length_epochs(raw, 2.0, preload=True, verbose="error")
    out = apply_ica(epochs, ica, exclude=[0])
    assert isinstance(out, mne.BaseEpochs)
    apply_ica(raw, ica, exclude=[0])
    np.testing.assert_array_equal(raw.get_data(), before)


@pytest.mark.parametrize("exclude, exc", [(0, TypeError), ("0", TypeError), ([1.0], TypeError),
                                          ([True], TypeError), ([0, 0], ValueError),
                                          ([N_COMP], ValueError), ([-1], ValueError)])
def test_apply_invalid_exclude(data, ica, exclude, exc):
    with pytest.raises(exc):
        apply_ica(data[0], ica, exclude=exclude)
