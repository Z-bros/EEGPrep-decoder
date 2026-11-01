"""Optional fixed features: reference parity, analytic values, alignment, independence."""

import subprocess
import sys

import mne
import numpy as np
import pytest
import scipy.signal as ss
from mne.time_frequency import psd_array_welch, tfr_array_morlet
from scipy import stats as sps

from eeg_decoder.features import (
    CLASSIC_BANDS,
    FeatureArray,
    band_power,
    hjorth,
    morlet_tfr,
    time_stats,
    welch_psd,
)
from eeg_decoder.io import to_raw

SF = 250.0
T = np.arange(int(2 * SF)) / SF  # 2 s epochs
A_CZ, A_PZ = 20e-6, 10e-6


def _epochs(n=6, bads=(), seed=0):
    """Cz = 20 uV 10 Hz sine, Pz = 10 uV 20 Hz sine, Oz = 5 uV white noise; two conditions."""
    rng = np.random.default_rng(seed)
    data = np.stack([
        np.vstack([A_CZ * np.sin(2 * np.pi * 10 * T), A_PZ * np.sin(2 * np.pi * 20 * T),
                   rng.normal(0, 5e-6, T.size)])
        for _ in range(n)
    ])
    info = mne.create_info(["Cz", "Pz", "Oz"], SF, "eeg")
    info["bads"] = list(bads)
    events = np.column_stack([np.arange(n) * 1000, np.zeros(n, int), np.tile([1, 2], n)[:n]])
    return mne.EpochsArray(data, info, events=events, event_id={"left": 1, "right": 2},
                           verbose="error")


# --- FeatureArray -----------------------------------------------------------


def test_feature_array_alignment_immutability_and_flatten():
    ep = _epochs()
    fa = band_power(ep, {"alpha": (8, 13), "beta": (13, 30)}, window_s=1.0)
    assert isinstance(fa, FeatureArray)
    assert fa.dims == ("epoch", "channel", "band") and fa.shape == (6, 3, 2)
    assert fa.coords == {"channel": ("Cz", "Pz", "Oz"), "band": ("alpha", "beta")}
    np.testing.assert_array_equal(fa.events, ep.events)
    assert dict(fa.event_id) == {"left": 1, "right": 2}
    with pytest.raises(ValueError):
        fa.data[0, 0, 0] = 1.0  # read-only
    with pytest.raises(ValueError):
        fa.events[0, 2] = 9
    with pytest.raises(TypeError):
        fa.coords["band"] = ()
    X, names = fa.flatten()
    assert X.shape == (6, 6)
    assert names == ["Cz:alpha", "Cz:beta", "Pz:alpha", "Pz:beta", "Oz:alpha", "Oz:beta"]
    np.testing.assert_array_equal(X[:, 3], fa.data[:, 1, 1])  # channel-major columns
    X[0, 0] = 0.0  # flatten returns a writable copy
    assert fa.data[0, 0, 0] != 0.0
    assert "band: 2" in repr(fa)


# --- welch_psd ---------------------------------------------------------------


def test_welch_psd_matches_scipy_and_mne():
    ep = _epochs()
    fa = welch_psd(ep, window_s=1.0, window="hamming", overlap=0.5)
    nper = 250
    f_ref, p_ref = ss.welch(ep.get_data(), fs=SF, window="hamming", nperseg=nper,
                            noverlap=nper // 2, axis=-1)
    np.testing.assert_allclose(fa.data, p_ref, rtol=1e-12)
    np.testing.assert_allclose(fa.coords["frequency"], f_ref)
    p_mne, _ = psd_array_welch(ep.get_data(), SF, n_fft=nper, n_per_seg=nper,
                               n_overlap=nper // 2, window="hamming", verbose="error")
    # Bins without signal hold ~1e-41 V^2/Hz of round-off; compare on the spectrum's scale.
    np.testing.assert_allclose(fa.data, p_mne, rtol=1e-10, atol=1e-12 * p_mne.max())
    assert fa.units == "V^2/Hz"


def test_welch_psd_resolution_peaks_and_range():
    fa = welch_psd(_epochs(), window_s=2.0, fmin=5, fmax=40)
    f = np.array(fa.coords["frequency"])
    assert f[1] - f[0] == pytest.approx(0.5) and f.min() >= 5 and f.max() <= 40
    assert f[fa.data[0, 0].argmax()] == 10.0 and f[fa.data[0, 1].argmax()] == 20.0


@pytest.mark.parametrize(
    "kwargs, exc",
    [({"window_s": 3.0}, ValueError), ({"window_s": 0}, ValueError), ({"window_s": "1"}, TypeError),
     ({"window_s": 1.0, "overlap": 1.0}, ValueError), ({"window_s": 1.0, "window": "kaiser"}, ValueError),
     ({"window_s": 1.0, "fmin": 40, "fmax": 10}, ValueError), ({"window_s": 1.0, "fmax": 200}, ValueError),
     ({"window_s": 1.0, "include_bads": "no"}, TypeError)],
)
def test_welch_psd_invalid(kwargs, exc):
    with pytest.raises(exc):
        welch_psd(_epochs(), **kwargs)


def test_window_s_is_required():
    with pytest.raises(TypeError):
        welch_psd(_epochs())
    with pytest.raises(TypeError):
        band_power(_epochs(), CLASSIC_BANDS)


# --- band_power -------------------------------------------------------------


def test_band_power_of_a_sine_is_half_amplitude_squared():
    fa = band_power(_epochs(), CLASSIC_BANDS, window_s=1.0)
    alpha, beta = fa.coords["band"].index("alpha"), fa.coords["band"].index("beta")
    assert fa.data[0, 0, alpha] == pytest.approx(A_CZ ** 2 / 2, rel=1e-3)
    assert fa.data[0, 1, beta] == pytest.approx(A_PZ ** 2 / 2, rel=1e-3)
    assert fa.units == "V^2"


def test_tiling_bands_add_up_and_relative_sums_to_one():
    tiles = {"a": (1, 8), "b": (8, 20), "c": (20, 45)}
    ep = _epochs()
    absolute = band_power(ep, tiles, window_s=1.0)
    whole = band_power(ep, {"all": (1, 45)}, window_s=1.0)
    np.testing.assert_allclose(absolute.data.sum(-1), whole.data[..., 0], rtol=1e-12)
    rel = band_power(ep, tiles, window_s=1.0, relative=True, total_range=(1, 45))
    np.testing.assert_allclose(rel.data.sum(-1), 1.0, rtol=1e-12)
    assert rel.units == "fraction"


def test_band_power_is_psd_summed_over_half_open_bins():
    # Definition: sum of PSD bins with low <= f < high, times the bin width.
    # The 10 Hz bin on the shared edge belongs to [10, 12) only.
    ep = _epochs()
    psd = welch_psd(ep, window_s=1.0)
    f = np.array(psd.coords["frequency"])
    fa = band_power(ep, {"lo": (8, 10), "hi": (10, 12)}, window_s=1.0)
    for i, (lo, hi) in enumerate([(8, 10), (10, 12)]):
        manual = psd.data[..., (f >= lo) & (f < hi)].sum(-1) * (f[1] - f[0])
        np.testing.assert_allclose(fa.data[..., i], manual, rtol=1e-12)
    assert fa.data[0, 0, 1] > fa.data[0, 0, 0]  # the 10 Hz peak bin is in "hi"


def test_log_band_power_in_db_re_1_uv2():
    fa = band_power(_epochs(), {"alpha": (8, 13)}, window_s=1.0, log=True)
    expected = 10 * np.log10((A_CZ ** 2 / 2) / 1e-12)  # 20 uV sine: 200 uV^2 -> 23 dB
    assert fa.data[0, 0, 0] == pytest.approx(expected, abs=0.01)
    assert fa.units == "dB re 1 uV^2"


@pytest.mark.parametrize(
    "bands, kwargs, exc, match",
    [({}, {}, TypeError, "non-empty"), ([(8, 13)], {}, TypeError, "mapping"),
     ({"a": (13, 8)}, {}, ValueError, "below"), ({"a": (8, 200)}, {}, ValueError, "Nyquist"),
     ({"a": (10, 11)}, {}, ValueError, "two frequency bins"),
     ({"a": (8, 13), "b": (12, 20)}, {}, ValueError, "overlap"),
     ({"a": 8}, {}, TypeError, "pair"), ({1: (8, 13)}, {}, TypeError, "strings"),
     ({"a": (8, 13)}, {"relative": True}, ValueError, "total_range"),
     ({"a": (8, 13)}, {"total_range": (1, 45)}, ValueError, "only used"),
     ({"a": (8, 13)}, {"relative": True, "log": True, "total_range": (1, 45)}, ValueError, "not both")],
)
def test_band_power_invalid(bands, kwargs, exc, match):
    with pytest.raises(exc, match=match):
        band_power(_epochs(), bands, window_s=1.0, **kwargs)


def test_overlap_allowed_explicitly():
    fa = band_power(_epochs(), {"a": (8, 13), "b": (12, 20)}, window_s=1.0, allow_overlap=True)
    assert fa.shape[-1] == 2


def test_classic_bands_is_an_immutable_example():
    assert CLASSIC_BANDS["alpha"] == (8.0, 13.0)
    with pytest.raises(TypeError):
        CLASSIC_BANDS["alpha"] = (8, 12)


# --- time_stats and hjorth --------------------------------------------------


def test_time_stats_match_numpy_and_scipy():
    ep = _epochs()
    fa = time_stats(ep)
    x = ep.get_data()
    ref = {"mean": x.mean(-1), "var": x.var(-1), "std": x.std(-1), "ptp": np.ptp(x, -1),
           "rms": np.sqrt((x ** 2).mean(-1)), "skew": sps.skew(x, axis=-1),
           "kurtosis": sps.kurtosis(x, axis=-1)}
    for i, name in enumerate(fa.coords["stat"]):
        np.testing.assert_allclose(fa.data[..., i], ref[name], rtol=1e-12, atol=1e-30)
    assert "var=V^2" in fa.units and "kurtosis=unitless" in fa.units


def test_time_stats_subset_and_invalid():
    assert time_stats(_epochs(), ["ptp", "var"]).coords["stat"] == ("ptp", "var")
    for stats, exc in [("var", TypeError), ([], ValueError), (["median"], ValueError),
                       (["var", "var"], ValueError)]:
        with pytest.raises(exc):
            time_stats(_epochs(), stats)


def test_hjorth_of_a_sine_matches_theory():
    fa = hjorth(_epochs())
    activity, mobility, complexity = (fa.data[0, 0, i] for i in range(3))
    assert activity == pytest.approx(A_CZ ** 2 / 2, rel=1e-3)
    exact = 2 * SF * np.sin(np.pi * 10 / SF)  # discrete-difference mobility, ~2*pi*10
    assert mobility == pytest.approx(exact, rel=5e-3)
    assert mobility == pytest.approx(2 * np.pi * 10, rel=5e-3)
    assert complexity == pytest.approx(1.0, abs=0.01)
    assert fa.coords["parameter"] == ("activity", "mobility", "complexity")


def test_hjorth_flat_channel_gives_nan_and_noise_inflates_mobility():
    ep = _epochs()
    data = ep.get_data()
    data[:, 1] = 0.0
    flat = hjorth(mne.EpochsArray(data, ep.info, verbose="error"))
    assert np.isnan(flat.data[0, 1, 1]) and np.isnan(flat.data[0, 1, 2])
    assert hjorth(ep).data[0, 2, 1] > 10 * hjorth(ep).data[0, 0, 1] / 2  # white noise >> 10 Hz sine


# --- morlet_tfr -------------------------------------------------------------


def test_morlet_tfr_matches_mne_and_peaks_at_signal_frequency():
    ep = _epochs()
    freqs = np.arange(6.0, 31.0, 2.0)
    fa = morlet_tfr(ep, freqs, n_cycles=freqs / 2)
    ref = tfr_array_morlet(ep.get_data(), SF, freqs, n_cycles=freqs / 2, output="power",
                           verbose="error")
    np.testing.assert_allclose(fa.data, ref, rtol=1e-12)
    assert fa.dims == ("epoch", "channel", "frequency", "time") and fa.units == "V^2"
    assert freqs[fa.data[0, 0].mean(-1).argmax()] == 10.0
    assert freqs[fa.data[0, 1].mean(-1).argmax()] == 20.0
    np.testing.assert_allclose(fa.coords["time"], ep.times)


def test_morlet_tfr_decim_thins_time_axis():
    ep = _epochs()
    fa = morlet_tfr(ep, [10.0, 20.0], n_cycles=5, decim=5)
    assert fa.shape[-1] == len(ep.times[::5])
    np.testing.assert_allclose(fa.coords["time"], ep.times[::5])


@pytest.mark.parametrize(
    "freqs, kwargs, exc",
    [([], {"n_cycles": 5}, TypeError), ([0.0, 10.0], {"n_cycles": 5}, ValueError),
     ([10.0, 130.0], {"n_cycles": 5}, ValueError), ([20.0, 10.0], {"n_cycles": 5}, ValueError),
     ([10.0, 20.0], {"n_cycles": [5, 5, 5]}, TypeError), ([10.0], {"n_cycles": 0}, ValueError),
     ([10.0], {"n_cycles": 5, "decim": 0}, ValueError), ("10", {"n_cycles": 5}, TypeError),
     ([2.0], {"n_cycles": 7}, ValueError)],  # last: wavelet longer than the 2 s epoch (MNE)
)
def test_morlet_tfr_invalid(freqs, kwargs, exc):
    with pytest.raises(exc):
        morlet_tfr(_epochs(), freqs, **kwargs)


# --- shared behaviour -------------------------------------------------------


@pytest.mark.parametrize("fn", [
    lambda e, **k: welch_psd(e, window_s=1.0, **k),
    lambda e, **k: band_power(e, CLASSIC_BANDS, window_s=1.0, **k),
    lambda e, **k: time_stats(e, **k),
    lambda e, **k: hjorth(e, **k),
    lambda e, **k: morlet_tfr(e, [10.0], n_cycles=5, **k),
])
def test_bads_excluded_by_default_inputs_untouched_deterministic(fn):
    ep = _epochs(bads=["Oz"])
    before = ep.get_data().copy()
    out = fn(ep)
    assert out.coords["channel"] == ("Cz", "Pz")
    assert fn(ep, include_bads=True).coords["channel"] == ("Cz", "Pz", "Oz")
    np.testing.assert_array_equal(ep.get_data(), before)
    np.testing.assert_array_equal(fn(ep).data, out.data)


def test_invalid_inputs_shared():
    ep = _epochs()
    raw = to_raw(np.zeros((2, 500)), SF, ["Cz", "Pz"])
    with pytest.raises(TypeError, match="per epoch"):
        hjorth(raw)
    with pytest.raises(TypeError, match="EpochsArray"):
        hjorth(ep.get_data())
    nan = ep.get_data()
    nan[0, 0, 5] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        hjorth(mne.EpochsArray(nan, ep.info, verbose="error"))
    eog = mne.EpochsArray(np.zeros((2, 1, 500)), mne.create_info(["EOG"], SF, "eog"), verbose="error")
    with pytest.raises(ValueError, match="No EEG"):
        hjorth(eog)
    empty = _epochs()
    empty.drop(list(range(len(empty))), verbose="error")
    with pytest.raises(ValueError, match="no epochs"):
        hjorth(empty)


# --- independence from the preprocessing pipeline ---------------------------


def test_features_are_not_pipeline_steps():
    from eeg_decoder.preprocessing import PreprocessingConfig

    for op in ("band_power", "welch_psd", "hjorth", "time_stats", "morlet_tfr"):
        with pytest.raises(ValueError, match="unknown op"):
            PreprocessingConfig([{"op": op}])


def test_preparing_data_never_imports_features():
    code = ("import sys, eeg_decoder.preprocessing, eeg_decoder.epoching, "
            "eeg_decoder.artifacts, eeg_decoder.quality; "
            "print('eeg_decoder.features' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
