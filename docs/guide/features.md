# Optional feature extraction

Fixed, non-learned descriptors for users who want them. **Not needed to
prepare data and not an `EEGPipeline` step**; importing the preprocessing
modules never loads them. Each value uses only one epoch's own samples, so
features are leakage-free. Supervised transforms such as CSP are out of scope.

```python
from eeg_decoder.features import (
    CLASSIC_BANDS, welch_psd, band_power, time_stats, hjorth, morlet_tfr,
)

psd = welch_psd(epochs, window_s=2.0)                 # 0.5 Hz resolution, V^2/Hz
bp = band_power(epochs, CLASSIC_BANDS, window_s=2.0)  # V^2; bands always explicit
rel = band_power(epochs, {"alpha": (8, 13)}, window_s=2.0,
                 relative=True, total_range=(1, 45))
h = hjorth(epochs)                                    # low-pass first: noise-sensitive
tfr = morlet_tfr(epochs, freqs=np.arange(4, 41, 2), n_cycles=np.arange(4, 41, 2) / 2)

X, names = bp.flatten()        # (n_epochs, n_features), names like "C3:alpha"
y = bp.events[:, 2]            # rows align with epochs.events
```

- Every function returns a `FeatureArray`: a read-only array with named axes
  (`dims`, e.g. `("epoch", "channel", "band")`), axis labels (`coords`),
  `units`, and the epochs' `events` / `event_id`, so features never lose
  their labels. `flatten()` gives a 2-D table for downstream tools.
- **Nothing is left to library defaults.** `window_s` (frequency resolution
  `1 / window_s` Hz) is required, because SciPy and MNE default to different
  windows and segment lengths, and MNE's `compute_psd` defaults to
  multitaper. Bands are half-open `[low, high)`, must span two frequency bins,
  and may not overlap unless `allow_overlap=True`. `CLASSIC_BANDS` (delta
  1-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-45 Hz) is an example you
  pass explicitly.
- Band power is the PSD summed over the bins in the band times the bin width,
  so bands that tile a range add up exactly; `relative=True` divides by an
  explicit `total_range`, and `log=True` gives dB re 1 µV².
- Hjorth mobility is in rad/s (≈ 2πf for a sine at f Hz) and complexity is 1
  for a pure sine; broadband noise inflates both strongly, so low-pass first.
- Morlet power has no baseline normalisation. MNE's wavelets are longer than
  `n_cycles / f` (7 cycles at 10 Hz needs ~1.1 s); MNE raises if one does not
  fit the epoch, and values within half a wavelet of the epoch edges carry
  edge effects.
- Input is `mne.Epochs`. For NumPy arrays, build `mne.EpochsArray(data, info)`
  with explicit `sfreq` and channel names first.
