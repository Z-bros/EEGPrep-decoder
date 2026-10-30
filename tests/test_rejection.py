"""Peak-to-peak epoch rejection on synthetic epochs with planted artifacts."""

import mne
import numpy as np
import pytest

from eeg_decoder.artifacts import reject_epochs

SFREQ = 250.0
N_EP, N_SAMP = 6, 176
CH = ["Fz", "Cz", "Pz", "EOG", "STI"]
TYPES = ["eeg", "eeg", "eeg", "eog", "stim"]
BLINK_EPOCH, FLAT_EPOCH, EOG_EPOCH = 1, 3, 4


def _epochs(bads=()):
    """Clean ~10 uV noise, with one EEG artifact, one flat EEG trace, one big EOG."""
    rng = np.random.default_rng(11)
    data = rng.normal(0, 10e-6, (N_EP, len(CH), N_SAMP))
    data[:, 4] = 0.0
    data[BLINK_EPOCH, 0, 80:100] += 300e-6  # 300 uV deflection on Fz
    data[FLAT_EPOCH, 2] = 1e-7  # flat (disconnected) Pz
    data[EOG_EPOCH, 3, 50:70] += 400e-6  # blink on EOG only
    info = mne.create_info(CH, SFREQ, TYPES)
    info["bads"] = list(bads)
    events = np.column_stack([np.arange(N_EP) * 1000, np.zeros(N_EP, int), np.ones(N_EP, int)])
    return mne.EpochsArray(data, info, events=events, tmin=-0.2, baseline=None, verbose="error")


def _kept(ep):
    return [i for i, log in enumerate(ep.drop_log) if not log]


def test_reject_drops_large_artifact_and_logs_channel():
    out = reject_epochs(_epochs(), reject={"eeg": 150e-6})
    assert _kept(out) == [0, 2, 3, 4, 5]
    assert out.drop_log[BLINK_EPOCH] == ("Fz",)


def test_flat_drops_dead_channel_epoch():
    out = reject_epochs(_epochs(), flat={"eeg": 1e-6})
    assert _kept(out) == [0, 1, 2, 4, 5]
    assert out.drop_log[FLAT_EPOCH] == ("Pz",)


def test_thresholds_are_per_channel_type():
    assert len(reject_epochs(_epochs(), reject={"eeg": 150e-6})) == 5  # EOG blink ignored
    out = reject_epochs(_epochs(), reject={"eeg": 150e-6, "eog": 250e-6}, flat={"eeg": 1e-6})
    assert _kept(out) == [0, 2, 5]


def test_matches_mne_drop_bad():
    crit = dict(reject={"eeg": 150e-6, "eog": 250e-6}, flat={"eeg": 1e-6})
    ours = reject_epochs(_epochs(), **crit)
    ref = _epochs().drop_bad(verbose="error", **crit)
    np.testing.assert_array_equal(ours.get_data(), ref.get_data())
    assert ours.drop_log == ref.drop_log


def test_bad_channels_are_ignored():
    out = reject_epochs(_epochs(bads=["Fz", "Pz"]), reject={"eeg": 150e-6}, flat={"eeg": 1e-6})
    assert len(out) == N_EP  # artifact on Fz and flat Pz both on bad channels


def test_input_untouched():
    ep = _epochs()
    before = ep.get_data().copy()
    out = reject_epochs(ep, reject={"eeg": 150e-6})
    assert out is not ep and len(ep) == N_EP
    np.testing.assert_array_equal(ep.get_data(), before)


def test_all_rejected_raises_with_culprits():
    with pytest.raises(ValueError, match=r"All 6 epochs.*Fz"):
        reject_epochs(_epochs(), reject={"eeg": 1e-6})


def test_microvolts_passed_as_volts_rejected():
    with pytest.raises(ValueError, match="150e-6"):
        reject_epochs(_epochs(), reject={"eeg": 150})


@pytest.mark.parametrize(
    "kwargs, exc",
    [({}, ValueError), ({"reject": {}}, TypeError), ({"reject": [150e-6]}, TypeError),
     ({"reject": {"eeg": 0}}, ValueError), ({"reject": {"eeg": -1e-6}}, ValueError),
     ({"reject": {"eeg": np.nan}}, ValueError), ({"reject": {"eeg": "1e-4"}}, TypeError),
     ({"reject": {"ecg": 1e-3}}, ValueError), ({"flat": {"stim": 1e-6, "eeg": 1e-6}}, ValueError)],
)
def test_invalid_thresholds(kwargs, exc):
    with pytest.raises(exc):
        reject_epochs(_epochs(), **kwargs)


def test_threshold_for_type_with_only_bad_channels_raises():
    with pytest.raises(ValueError, match="no\\s+good channels"):
        reject_epochs(_epochs(bads=["EOG"]), reject={"eog": 250e-6})


def test_rejects_continuous_and_empty():
    raw = mne.io.RawArray(np.zeros((1, 100)), mne.create_info(["Cz"], SFREQ, "eeg"), verbose="error")
    with pytest.raises(TypeError, match="Epochs"):
        reject_epochs(raw, reject={"eeg": 1e-4})
    empty = _epochs()
    empty.drop(list(range(N_EP)), verbose="error")
    with pytest.raises(ValueError, match="no epochs"):
        reject_epochs(empty, reject={"eeg": 1e-4})
