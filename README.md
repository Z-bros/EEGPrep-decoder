# EEG_Decoder

A modular, reproducible Python toolkit for EEG preprocessing and decoding.
Reusable functions instead of notebook code, built on the scientific Python
stack (NumPy, SciPy, MNE-Python, scikit-learn).

> **Status: pre-alpha (Phase 2a — array adapter and filtering).** Available:
> `to_raw`, `bandpass`, `notch`. Re-referencing, resampling, epoching,
> artifact rejection, features, and decoding are developed incrementally and
> are not available yet.

## Installation

Requires Python 3.11 or newer.

```bash
git clone https://github.com/Z-bros/EEGPrep-decoder.git
cd EEGPrep-decoder
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Heavy dependencies (PyTorch, Braindecode, pyRiemann, MOABB) will be optional
extras so that preprocessing never requires the deep-learning stack.

## Quick check

```python
import eeg_decoder
from eeg_decoder import EEGConfig

print(eeg_decoder.__version__)
cfg = EEGConfig(seed=42)      # immutable; record it with every experiment
print(cfg.to_dict())          # {'seed': 42, 'float_dtype': 'float64'}
```

## Minimal preprocessing example

```python
import numpy as np
from eeg_decoder.io import to_raw
from eeg_decoder.preprocessing import bandpass, notch

data_uv = np.random.default_rng(0).normal(0, 10, size=(8, 2500))  # (channels, samples)
raw = to_raw(data_uv, sfreq=250.0,
             ch_names=["Fz", "Cz", "Pz", "C3", "C4", "O1", "O2", "Oz"],
             units="uV")                     # converted to volts, the MNE convention

raw = notch(raw, [50.0, 100.0])              # line noise: say 50 or 60 Hz explicitly
raw = bandpass(raw, l_freq=1.0, h_freq=40.0) # returns a new object each time
```

Functions take and return MNE objects (`mne.io.Raw`), so channel names,
types, sampling rate, and units travel with the signal. NumPy arrays enter
once, through `to_raw`.

### Input/output specification

| Function | Input | Output | Notes |
|---|---|---|---|
| `io.to_raw` | array `(n_channels, n_samples)`, `sfreq`, `ch_names`, `ch_types`, `units` | `mne.io.RawArray`; voltage channels in volts | Rejects empty arrays, transposed-looking input, NaN/Inf, duplicate names, and (optionally) voltage channels above 10 mV. Stim/misc channels are never rescaled |
| `preprocessing.bandpass` | continuous `mne.io.BaseRaw`, `l_freq`, `h_freq` (either may be `None`) | filtered copy | Requires `0 < l_freq < h_freq < sfreq/2` and a recording longer than the filter; rejects `Epochs` |
| `preprocessing.notch` | continuous `mne.io.BaseRaw`, explicit `freqs` | filtered copy | Frequencies at or above Nyquist raise, never silently dropped; recording must be longer than the filter |

**Which channels are filtered.** With the default `picks=None`, both filters
process EEG channels, *including those marked bad*, and leave EOG, ECG, EMG,
stim and misc channels untouched (verified on MNE 1.13). Use
`picks=["eeg", "eog"]` to filter EOG too, and never include stim channels in
`picks`. `raw.info["highpass"]`/`["lowpass"]` are updated only when all EEG
channels are band-passed; notch filtering is never recorded there.

**Minimum recording length.** MNE only warns when an FIR filter is longer
than the signal; here that raises `ValueError`. At 250 Hz a 1-40 Hz
band-pass needs 825 samples (3.3 s) and a 50 Hz notch 1651 samples (6.6 s).

**The 10 mV check is a heuristic.** It catches microvolt data passed as volts
(20 µV would appear as 20 V); it is not a physiological bound. Data with large
electrode DC offsets, intracranial recordings, or saturated amplifiers can
legitimately exceed it: pass `check_amplitude=False`. The check never alters
values.

### Scientific assumptions

- **Zero-phase FIR filters** (MNE `firwin`, Hamming window): no time shift
  of waveforms, so ERP latencies are preserved. Non-causal, so they are for
  offline analysis, not real-time decoding.
- **Cutoffs are passband edges, not -3 dB points.** With MNE's automatic
  transition bands, `l_freq=1` attenuates progressively over 0-1 Hz (-6 dB at
  0.5 Hz), so slow drift *inside* that band is reduced but not removed;
  `h_freq=40` reaches full attenuation (~53 dB) by 50 Hz.
- **Filter continuous data before epoching.** Filtering short epochs puts the
  filter's edge transient inside every trial, so the filters reject `Epochs`.
- **Line-noise frequency is never assumed.** 50 Hz (Asia, Europe, Africa) or
  60 Hz (Americas) must be passed explicitly; harmonics are not added for you.
- **A notch is redundant when the line frequency is in the band-pass
  stopband**, i.e. at or above `h_freq` plus its transition band. Measured at
  250 Hz: band-pass 1-40 Hz already attenuates 50 Hz by ~50 dB, but 1-45 Hz
  only by ~4.5 dB (50 Hz falls in the 45-56 Hz transition band). Keep the notch
  when analysing frequencies near or above the line frequency, when `h_freq`
  sits within one transition width of it, or when line noise is so strong that
  ~50 dB is not enough. Otherwise it only adds a long filter and ringing.

## Conventions (fixed from the start)

- Continuous EEG: `(n_channels, n_samples)`. Epoched EEG:
  `(n_epochs, n_channels, n_samples)`. Axes are never inferred silently.
- Signal units: volts (SI), as in MNE-Python, unless a function documents otherwise.
- Inputs are never modified in place unless a function explicitly says so.
- Every stochastic operation takes an explicit seed.

## Development

```bash
pip install -e ".[dev]"
pytest
```

No real EEG dataset is needed: tests use seeded synthetic signals with known
spectral content. Do not commit EEG recordings (see `.gitignore`).

## Known limitations

Pre-alpha; the API will change.

- Only FIR filtering is exposed; IIR filters and custom transition bandwidths
  are not yet configurable.
- Notch filtering is not recorded in `raw.info` (band-pass cutoffs are).
  A full processing record arrives with the pipeline configuration (Phase 2d).
- No file readers yet: load recordings with MNE (e.g. `mne.io.read_raw_edf`)
  and pass the resulting `Raw` object in directly.

## License

MIT — see [LICENSE](LICENSE).
