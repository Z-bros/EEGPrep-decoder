import mne
import numpy as np
import pytest

from conftest import CH_NAMES, SFREQ
from eeg_decoder.io import to_raw


def test_returns_raw_with_metadata(continuous_eeg):
    d = continuous_eeg
    raw = to_raw(d.data, d.sfreq, d.ch_names)
    assert isinstance(raw, mne.io.BaseRaw)
    assert raw.info["sfreq"] == SFREQ
    assert raw.ch_names == list(CH_NAMES)
    assert set(raw.get_channel_types()) == {"eeg"}
    np.testing.assert_array_equal(raw.get_data(), d.data)


def test_input_is_copied_not_shared(continuous_eeg):
    original = continuous_eeg.data.copy()
    raw = to_raw(continuous_eeg.data, SFREQ, CH_NAMES)
    raw._data[:] = 0.0
    np.testing.assert_array_equal(continuous_eeg.data, original)


@pytest.mark.parametrize("units, scale", [("V", 1.0), ("mV", 1e-3), ("uV", 1e-6)])
def test_unit_conversion_to_volts(units, scale):
    data = np.full((2, 100), 20.0 if units == "uV" else 20e-6 / scale)
    raw = to_raw(data, 100.0, ["a", "b"], units=units)
    np.testing.assert_allclose(raw.get_data(), 20e-6)


def test_mixed_channel_types():
    raw = to_raw(np.zeros((3, 10)), 100.0, ["Fz", "EOG", "STI"], ["eeg", "eog", "stim"])
    assert raw.get_channel_types() == ["eeg", "eog", "stim"]


def test_microvolts_passed_as_volts_is_rejected(continuous_eeg):
    with pytest.raises(ValueError, match="units"):
        to_raw(continuous_eeg.data * 1e6, SFREQ, CH_NAMES)  # uV labelled as V
    # ...but can be allowed explicitly (e.g. DC-offset recordings).
    to_raw(np.full((1, 10), 0.05), SFREQ, ["Fz"], check_amplitude=False)


@pytest.mark.parametrize("shape", [(100,), (2, 3, 100)])
def test_wrong_dimensionality(shape):
    with pytest.raises(ValueError, match="2-D"):
        to_raw(np.zeros(shape), SFREQ, ["a", "b"])


def test_transposed_input_is_detected_not_guessed(continuous_eeg):
    with pytest.raises(ValueError, match="transposed"):
        to_raw(continuous_eeg.data.T, SFREQ, CH_NAMES)


def test_channel_name_problems():
    with pytest.raises(ValueError, match="does not match"):
        to_raw(np.zeros((3, 10)), SFREQ, ["a", "b"])
    with pytest.raises(ValueError, match="Duplicate"):
        to_raw(np.zeros((2, 10)), SFREQ, ["a", "a"])
    with pytest.raises(ValueError, match="ch_types"):
        to_raw(np.zeros((2, 10)), SFREQ, ["a", "b"], ["eeg"])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_values(bad):
    data = np.zeros((2, 10))
    data[1, 5] = bad
    with pytest.raises(ValueError, match="NaN or Inf"):
        to_raw(data, SFREQ, ["a", "b"])


@pytest.mark.parametrize("sfreq, exc", [(0, ValueError), (-250, ValueError),
                                        (np.nan, ValueError), (np.inf, ValueError),
                                        (True, TypeError), ("250", TypeError)])
def test_invalid_sfreq(sfreq, exc):
    with pytest.raises(exc):
        to_raw(np.zeros((2, 10)), sfreq, ["a", "b"])


def test_invalid_units_and_dtype():
    with pytest.raises(ValueError, match="units"):
        to_raw(np.zeros((2, 10)), SFREQ, ["a", "b"], units="microvolts")
    with pytest.raises(ValueError, match="real-valued"):
        to_raw(np.zeros((2, 10), dtype=complex), SFREQ, ["a", "b"])
