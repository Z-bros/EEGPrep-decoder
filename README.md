# EEGPrep-decoder

A reusable, **model-agnostic** Python library for EEG preprocessing and data
preparation, inspired by Med_Im_Prepper. Reusable, tested functions instead
of notebook code, built on NumPy, SciPy, and MNE-Python. The import name is
`eeg_decoder`.

It turns raw recordings into clean, documented, analysis-ready data for any
downstream use (ERP analysis, statistics, machine learning, deep learning),
and stops there.

> **Status: pre-alpha (Phases 1-2 complete).** Available: `to_raw`,
> `bandpass`, `notch`, `rereference`, `resample`, `make_epochs`,
> `baseline_correct`, `reject_epochs`, and the `EEGPipeline` /
> `PreprocessingConfig` that chains them. Next: artifact detection and
> correction. See the [roadmap](docs/ROADMAP.md).

## Scope

**In scope:** preprocessing, artifact detection and correction (including
bad-channel assessment and optional ICA), signal-quality assessment and
visualization, *optional* fixed feature extraction (PSD, band power, Hjorth,
time-frequency), reproducible EEG augmentation, dataset preparation with
metadata, and export of prepared arrays or MNE objects.

**Out of scope:** classification models, model training, fine-tuning,
knowledge distillation, and model evaluation frameworks. **Supervised
learned transformations such as CSP are not part of the library**; they
belong in your downstream modelling code (e.g. a scikit-learn `Pipeline`
with `mne.decoding.CSP`), fitted on training data only.

Feature extraction, when added, is optional and independent: preparing EEG
data never requires it.

## Installation

Requires Python 3.11 or newer.

```bash
git clone https://github.com/Z-bros/EEGPrep-decoder.git
cd EEGPrep-decoder
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The core install needs only NumPy and MNE-Python (which brings SciPy).
Capabilities with extra requirements (e.g. ICA methods that need
scikit-learn or `python-picard`, or extra export formats) will be optional
dependency groups, so preparing data never pulls in a modelling stack.

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
from eeg_decoder.preprocessing import bandpass, notch, rereference, resample

data_uv = np.random.default_rng(0).normal(0, 10, size=(8, 2500))  # (channels, samples)
raw = to_raw(data_uv, sfreq=250.0,
             ch_names=["Fz", "Cz", "Pz", "C3", "C4", "O1", "O2", "Oz"],
             units="uV")                     # converted to volts, the MNE convention

raw = notch(raw, [50.0, 100.0])              # line noise: say 50 or 60 Hz explicitly
raw = bandpass(raw, l_freq=1.0, h_freq=40.0) # returns a new object each time
raw = rereference(raw, "average")            # or e.g. ["M1", "M2"]; no default
raw = resample(raw, 125.0)                   # after low-pass; before epoching
```

Epoching, baseline correction, and rejection (continuing from `raw` above,
for a recording with a stim channel or annotations):

```python
import mne
from eeg_decoder.epoching import make_epochs, baseline_correct
from eeg_decoder.artifacts import reject_epochs

events = mne.find_events(raw)                # extract AFTER resampling
epochs = make_epochs(raw, events, {"left": 1, "right": 2}, tmin=-0.2, tmax=0.8)
epochs = baseline_correct(epochs, (None, 0.0))           # pre-stimulus window
epochs = reject_epochs(epochs, reject={"eeg": 150e-6},   # volts, fixed thresholds
                       flat={"eeg": 1e-6})
print(epochs.drop_log)                       # which events were dropped, and why
```

Recommended order: mark bad channels -> filter -> re-reference -> resample
-> epoch -> baseline -> reject. Each step returns a new object.

## Pipeline with a saved configuration

The same chain as a reproducible, shareable configuration. Each step is
`{"op": <function name>, **that function's keyword arguments}`, so the JSON
file reads exactly like the Python code above:

```python
from eeg_decoder.preprocessing import EEGPipeline, PreprocessingConfig

config = PreprocessingConfig([
    {"op": "notch", "freqs": [50.0, 100.0]},
    {"op": "bandpass", "l_freq": 1.0, "h_freq": 40.0},
    {"op": "rereference", "ref": "average"},
    {"op": "resample", "sfreq": 125.0},
    {"op": "make_epochs", "event_source": "stim", "stim_channel": "STI",
     "event_id": {"left": 1, "right": 2}, "tmin": -0.2, "tmax": 0.8},
    {"op": "baseline_correct", "baseline": [None, 0.0]},
    {"op": "reject_epochs", "reject": {"eeg": 150e-6}, "flat": {"eeg": 1e-6}},
])
config.save("preprocessing.json")            # share / version alongside results

epochs, record = EEGPipeline(config).run(raw, metadata={"subject": "01", "session": "T"})
```

- **Validated up front.** Unknown steps, misspelled parameters, values JSON
  cannot store (NaN, NumPy arrays), and impossible orders (e.g. a filter
  after `make_epochs`, or `baseline_correct` before it) raise when the config
  is built, before any data are touched. Values that depend on the data
  (e.g. a cutoff above Nyquist) are checked when the step runs, and the error
  names the step.
- **Events** are extracted inside the pipeline at the `make_epochs` step,
  after any resampling: `"event_source": "stim"` (with `stim_channel`) or
  `"annotations"` (annotation descriptions matched to `event_id` keys).
- **The record** is a JSON-serializable dict: the config, Python/package
  versions, your `metadata`, an input summary, and per step its parameters
  and output (sampling rate, filter edges, reference, epoch counts, epochs
  dropped). Running the same config on the same data gives an identical
  record. The recording's measurement date and subject info are deliberately
  left out, since they may be identifying.
- **No leakage.** No step learns anything from the data, so the pipeline can
  run on the whole dataset before a train/test split. Supervised transforms
  (e.g. CSP) and augmentation are deliberately not pipeline steps: apply them
  downstream, to training data only.
- Configs are immutable; `PreprocessingConfig.load(path)` and
  `config.to_dict()` give editable copies to derive variants from.

Functions take and return MNE objects (`mne.io.Raw`), so channel names,
types, sampling rate, and units travel with the signal. NumPy arrays enter
once, through `to_raw`.

### Input/output specification

| Function | Input | Output | Notes |
|---|---|---|---|
| `io.to_raw` | array `(n_channels, n_samples)`, `sfreq`, `ch_names`, `ch_types`, `units` | `mne.io.RawArray`; voltage channels in volts | Rejects empty arrays, transposed-looking input, NaN/Inf, duplicate names, and (optionally) voltage channels above 10 mV. Stim/misc channels are never rescaled |
| `preprocessing.bandpass` | continuous `mne.io.BaseRaw`, `l_freq`, `h_freq` (either may be `None`) | filtered copy | Requires `0 < l_freq < h_freq < sfreq/2` and a recording longer than the filter; rejects `Epochs` |
| `preprocessing.notch` | continuous `mne.io.BaseRaw`, explicit `freqs` | filtered copy | Frequencies at or above Nyquist raise, never silently dropped; recording must be longer than the filter |
| `preprocessing.rereference` | `mne.io.BaseRaw` or `mne.Epochs`, `ref` (`"average"` or EEG channel list), `allow_bads=False` | re-referenced copy | Only EEG changes; raises if bad EEG channels exist unless `allow_bads=True`; reference channels must exist, be EEG, and not be bad |
| `preprocessing.resample` | continuous `mne.io.BaseRaw`, `sfreq`, `method` (`"fft"`/`"polyphase"`) | resampled copy, `round(n * new/old)` samples | Anti-aliased; stim codes kept; raises if stim events would merge; rejects `Epochs` |
| `epoching.make_epochs` | continuous `mne.io.BaseRaw`, `events` `(n, 3)` int, `event_id` dict, `tmin`, `tmax`, `decim=1` | `mne.Epochs` `(n_epochs, n_channels, n_times)`, no baseline | Raises if a window runs outside the recording (MNE drops silently), on duplicate/unsorted/unknown events, or if `decim` would alias |
| `epoching.baseline_correct` | `mne.Epochs`, `baseline=(start, end)` in s | corrected copy | Applies to EEG *and* EOG/ECG/EMG, not stim; window must be inside the epoch |
| `artifacts.reject_epochs` | `mne.Epochs`, `reject`/`flat` dicts in volts per channel type | copy without rejected epochs; `drop_log` per event | Bad channels ignored; raises on thresholds > 10 mV (unit mistake) or if every epoch would be dropped |

**Epoching.** Event samples are absolute (they include `raw.first_samp`),
as `mne.find_events` returns them. Both `tmin` and `tmax` are included, so
an epoch has `round((tmax - tmin) * sfreq) + 1` samples. MNE silently drops
epochs that run past the edges of the recording; `make_epochs` raises
instead and lists those events. Epochs overlapping `BAD_*` annotations are
dropped and recorded in `epochs.drop_log`. No baseline is applied implicitly
(MNE's default is `(None, 0)`), and stored projectors are not applied.
`decim` must keep the new rate at least 3x `info["lowpass"]` (MNE's own
rule, which it only warns about).

**Rejection and leakage.** `reject_epochs` uses fixed peak-to-peak
thresholds chosen by you, so applying it before a train/test split leaks
nothing. Thresholds derived from the data (e.g. percentiles) would have to
be fitted on training folds only.

**Bad channels and re-referencing.** MNE leaves channels marked bad out of
the reference *and does not re-reference them*, so they stay in the old
reference (verified on MNE 1.13, for both average and channel references).
`rereference` therefore raises when bad EEG channels are present: interpolate
(`raw.interpolate_bads()`, needs electrode positions) or drop them first, or
pass `allow_bads=True` to accept the mixed result. The average reference is
only a good approximation of a neutral reference with many, evenly spread
electrodes.

**Resampling and events.** Stim-channel events move to the nearest new
sample (at most half a new sample of error: 5 ms at 100 Hz), and their codes
are kept. Events closer than one new sample would merge, so `resample` raises
instead of silently dropping one. Annotations are in seconds and are
unaffected. Extract events arrays *after* resampling. `"fft"` is an ideal
cut at the new Nyquist; `"polyphase"` rolls off earlier (a 45 Hz tone keeps
~93 % of its amplitude at 100 Hz). Low-pass first and the difference stops
mattering.

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
  offline analysis, not real-time (online) processing.
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

## Reproducibility

Save the `PreprocessingConfig` and the `record` from `EEGPipeline.run` with
your results (`json.dump(record, f, indent=2)`). Together they state what was
run, on what input, with which package versions, and what came out of each
step. Phase 2 contains no random operations, so the same config on the same
data always gives identical output.

## Development

```bash
pip install -e ".[dev]"
pytest
```

No real EEG dataset is needed: tests use seeded synthetic signals with known
spectral content. Do not commit EEG recordings (see `.gitignore`).

## Roadmap

Full details, design rules, and per-phase considerations:
[docs/ROADMAP.md](docs/ROADMAP.md).

| Phase | Topic | Status |
|---|---|---|
| 1 | Package foundation | Done |
| 2 | Core preprocessing and `EEGPipeline` | Done |
| 3 | Artifact detection and correction (bad channels, optional ICA) | Next |
| 4 | Signal-quality assessment and visualization | Planned |
| 5 | Optional fixed feature extraction (PSD, band power, Hjorth, time-frequency) | Planned |
| 6 | EEG data augmentation (seeded, event/label preserving) | Planned |
| 7 | Dataset preparation, metadata, and export | Planned |
| 8 | Documentation, benchmarking, and release | Planned |

## Known limitations

Pre-alpha; the API will change.

- Only FIR filtering is exposed; IIR filters and custom transition bandwidths
  are not yet configurable.
- When the functions are called directly, MNE's `raw.info` records band-pass
  cutoffs but not notch filtering, and only *that* the reference changed.
  Use `EEGPipeline`, whose record keeps every step's parameters.
- The pipeline runs one recording at a time; looping over subjects and
  storing records is up to the caller for now.
- No REST reference, and no restoring of an unrecorded online reference
  electrode (`mne.add_reference_channels`).
- No file readers yet: load recordings with MNE (e.g. `mne.io.read_raw_edf`)
  and pass the resulting `Raw` object in directly.

## License

MIT — see [LICENSE](LICENSE).
