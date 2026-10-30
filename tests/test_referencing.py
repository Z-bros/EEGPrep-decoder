"""Re-referencing: exact arithmetic on known signals, MNE parity, and edge cases."""

import mne
import numpy as np
import pytest

from eeg_decoder.io import to_raw
from eeg_decoder.preprocessing import rereference

SFREQ = 250.0
NAMES = ["Fz", "Cz", "Pz", "M1", "M2", "EOG", "STI"]
TYPES = ["eeg"] * 5 + ["eog", "stim"]
EEG = slice(0, 5)


def _raw(bads=()):
    rng = np.random.default_rng(7)
    data = rng.normal(0, 10e-6, (len(NAMES), 2500))
    data[EEG] += rng.normal(0, 30e-6, (5, 1))  # per-channel offsets: reference matters
    data[-1] = 0.0
    data[-1, [250, 1000, 1750]] = [1, 2, 1]
    raw = to_raw(data, SFREQ, NAMES, TYPES)
    raw.info["bads"] = list(bads)
    return raw


def _epochs(raw):
    events = mne.find_events(raw, "STI", verbose="error")
    return mne.Epochs(raw, events, tmin=-0.2, tmax=0.5, baseline=None,
                      preload=True, verbose="error")


# --- arithmetic -------------------------------------------------------------


def test_average_reference_subtracts_mean_of_good_eeg():
    raw = _raw()
    x = raw.get_data()
    out = rereference(raw, "average").get_data()
    np.testing.assert_allclose(out[EEG], x[EEG] - x[EEG].mean(axis=0), atol=1e-18)
    np.testing.assert_allclose(out[EEG].mean(axis=0), 0.0, atol=1e-18)


def test_single_channel_reference():
    raw = _raw()
    x = raw.get_data()
    out = rereference(raw, ["Cz"]).get_data()
    np.testing.assert_allclose(out[EEG], x[EEG] - x[1], atol=1e-18)
    assert np.all(out[1] == 0)  # reference channel kept, flat


def test_linked_mastoid_reference():
    raw = _raw()
    x = raw.get_data()
    out = rereference(raw, ["M1", "M2"]).get_data()
    np.testing.assert_allclose(out[EEG], x[EEG] - x[[3, 4]].mean(axis=0), atol=1e-18)
    np.testing.assert_allclose(out[3], -out[4], atol=1e-18)  # antisymmetric pair


@pytest.mark.parametrize("ref", ["average", ["Cz"], ["M1", "M2"]])
def test_matches_mne_directly(ref):
    raw = _raw()
    ours = rereference(raw, ref).get_data()
    ref_mne = raw.copy().set_eeg_reference(ref, projection=False, verbose="error").get_data()
    np.testing.assert_array_equal(ours, ref_mne)


# --- metadata, immutability, non-EEG channels -------------------------------


@pytest.mark.parametrize("ref", ["average", ["Cz"]])
def test_non_eeg_untouched_metadata_kept_input_unchanged(ref):
    raw = _raw()
    before = raw.get_data().copy()
    out = rereference(raw, ref)
    assert out is not raw
    np.testing.assert_array_equal(raw.get_data(), before)
    np.testing.assert_array_equal(out.get_data()[5:], before[5:])  # EOG + stim
    assert out.ch_names == raw.ch_names
    assert out.get_channel_types() == raw.get_channel_types()
    assert out.info["sfreq"] == raw.info["sfreq"]
    assert out.info["custom_ref_applied"] and not raw.info["custom_ref_applied"]
    assert out.info["projs"] == []  # applied directly, no deferred projector


def test_epochs_supported_and_order_independent():
    raw = _raw()
    ep_then_ref = rereference(_epochs(raw), "average")
    ref_then_ep = _epochs(rereference(raw, "average"))
    assert isinstance(ep_then_ref, mne.BaseEpochs)
    np.testing.assert_allclose(ep_then_ref.get_data(), ref_then_ep.get_data(), atol=1e-18)


def test_non_preloaded_epochs_are_loaded_on_the_copy():
    raw = _raw()
    events = mne.find_events(raw, "STI", verbose="error")
    lazy = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5, baseline=None, verbose="error")
    out = rereference(lazy, "average")
    assert out.preload and not lazy.preload


# --- bad channels -----------------------------------------------------------


@pytest.mark.parametrize("ref", ["average", ["Cz"]])
def test_bad_channels_rejected_by_default(ref):
    with pytest.raises(ValueError, match="allow_bads"):
        rereference(_raw(bads=["Pz"]), ref)


def test_allow_bads_follows_mne_and_leaves_bads_in_old_reference():
    raw = _raw(bads=["Pz"])
    x = raw.get_data()
    out = rereference(raw, "average", allow_bads=True)
    y = out.get_data()
    good = [0, 1, 3, 4]
    np.testing.assert_allclose(y[good], x[good] - x[good].mean(axis=0), atol=1e-18)
    np.testing.assert_array_equal(y[2], x[2])  # bad channel unchanged
    assert out.info["bads"] == ["Pz"]


def test_non_eeg_bads_do_not_block():
    rereference(_raw(bads=["EOG"]), "average")


# --- invalid inputs ---------------------------------------------------------


@pytest.mark.parametrize(
    "ref, match",
    [("Cz", r"ref=\['Cz'\]"), ("avg", "average"), ([], "at least one"),
     (["Xx"], "not found"), (["EOG"], "not EEG"), (["STI"], "not EEG"),
     (["Cz", "Cz"], "Duplicate")],
)
def test_invalid_ref_values(ref, match):
    with pytest.raises(ValueError, match=match):
        rereference(_raw(), ref)


def test_bad_reference_channel_rejected_even_with_allow_bads():
    with pytest.raises(ValueError, match="marked bad"):
        rereference(_raw(bads=["Cz"]), ["Cz"], allow_bads=True)


@pytest.mark.parametrize("ref", [None, 3, [1, 2]])
def test_invalid_ref_types(ref):
    with pytest.raises(TypeError):
        rereference(_raw(), ref)


def test_average_needs_two_good_eeg_channels():
    raw = to_raw(np.zeros((2, 100)), SFREQ, ["Cz", "EOG"], ["eeg", "eog"])
    with pytest.raises(ValueError, match="at least 2"):
        rereference(raw, "average")


def test_no_eeg_channels():
    raw = to_raw(np.zeros((2, 100)), SFREQ, ["EOG", "ECG"], ["eog", "ecg"])
    with pytest.raises(ValueError, match="No EEG"):
        rereference(raw, "average")


def test_empty_inputs():
    info = mne.create_info(["Fz", "Cz"], SFREQ, "eeg")
    with pytest.raises(ValueError, match="empty"):
        rereference(mne.io.RawArray(np.zeros((2, 0)), info, verbose="error"), "average")
    empty_ep = mne.EpochsArray(np.zeros((1, 2, 10)), info, verbose="error")
    empty_ep.drop([0], verbose="error")
    with pytest.raises(ValueError, match="no epochs"):
        rereference(empty_ep, "average")


def test_rejects_arrays():
    with pytest.raises(TypeError, match="to_raw"):
        rereference(np.zeros((2, 100)), "average")
