# EEGPrep-decoder

A reusable, **model-agnostic** Python library for EEG preprocessing and data
preparation, inspired by Med_Im_Prepper. Reusable, tested functions instead
of notebook code, built on NumPy, SciPy, and MNE-Python. The import name is
`eeg_decoder`.

It turns raw recordings into clean, documented, analysis-ready data for any
downstream use (ERP analysis, statistics, machine learning, deep learning),
and stops there.

> **Status: pre-alpha (Phases 1-5 complete).** Available: `to_raw`,
> `bandpass`, `notch`, `rereference`, `resample`, `make_epochs`,
> `baseline_correct`, `reject_epochs`, the `EEGPipeline` /
> `PreprocessingConfig` that chains them, artifact handling
> (`assess_channels`, `mark_bad_channels`, `interpolate_bad_channels`,
> `annotate_artifacts`, `add_annotations`, and optional ICA via `fit_ica`,
> `find_artifact_components`, `apply_ica`), signal-quality reports and
> plots (`recording_quality`, `epochs_quality`, `plot_psd_comparison`,
> `plot_channel_quality`, `plot_rejection_summary`), and optional features
> (`welch_psd`, `band_power`, `time_stats`, `hjorth`, `morlet_tfr`). Next:
> EEG data augmentation. See the [roadmap](docs/ROADMAP.md).

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

## Artifact handling

Find problems first, review the evidence, then act. Run these on continuous
data after a high-pass filter and **before** re-referencing to the average.

```python
from eeg_decoder.artifacts import (
    assess_channels, mark_bad_channels, interpolate_bad_channels,
    annotate_artifacts, add_annotations,
)

report = assess_channels(raw)                # evidence only; data untouched
print(report.summary())                      # e.g. "T7: deviation", "O1: correlation"
raw = mark_bad_channels(raw, report.bad)     # or a list you reviewed yourself

raw = raw.copy().set_montage("colin27_1020") # positions needed for interpolation
raw = interpolate_bad_channels(raw)          # only bad EEG channels change

bad_spans = annotate_artifacts(raw, peak={"eeg": 150e-6}, flat={"eeg": 1e-6})
raw = add_annotations(raw, bad_spans)        # make_epochs drops overlapping epochs
```

- **`assess_channels`** checks every EEG channel with five criteria adapted
  from the PREP pipeline (Bigdely-Shamlo et al., 2015) and its default
  thresholds: `nan`, `flat`, `deviation` (robust amplitude outlier, too large
  *or* too small), `correlation` (poorly correlated with every other channel
  in > 1 % of 1 s windows), and `hf_noise` (outlying share of power above
  50 Hz; skipped, with the reason, when the data hold none). Robust z-scores
  use the median and IQR, so bad channels cannot hide each other; at least 3
  EEG channels are required and more than ~16 are recommended. The result is
  an immutable, JSON-serializable `ChannelAssessment` with the reasons and
  per-channel values.
- **`interpolate_bad_channels`** uses MNE's spherical splines. It refuses
  when an EEG channel has no position or when more than `max_fraction`
  (default 20 %) of EEG channels are bad (MNE itself would interpolate 12 of
  16). Bad non-EEG channels (e.g. EOG) keep their bad mark; MNE alone would
  silently drop it. It is also available as the pipeline step
  `{"op": "interpolate_bad_channels"}`; assessment deliberately is not, so a
  person reviews it.
- **`annotate_artifacts`** marks spans as `BAD_peak` / `BAD_flat` using a
  sliding-window peak-to-peak amplitude in volts per channel type, the same
  quantity `reject_epochs` uses per epoch. It is used instead of MNE's
  `annotate_amplitude`, which compares consecutive samples and misses a
  smooth 300 µV blink entirely. Bad channels are ignored, so mark them first.
- All of these work per recording without labels, so they cause no leakage.

### Optional ICA (blinks, eye movements, heartbeat)

```python
from eeg_decoder.artifacts import fit_ica, find_artifact_components, apply_ica

ica = fit_ica(raw, n_components=20, seed=42)          # fit on a 1 Hz high-passed copy
report = find_artifact_components(ica, raw, eog=["EOG"], ecg="ECG")
print(report.to_dict())                               # proposed indices + correlations
raw = apply_ica(raw, ica, exclude=report.exclude)     # removes exactly these components
```

- Three explicit steps: fit (seeded, label-free), *propose* components by
  correlation with EOG/ECG channels, then remove exactly the indices you
  pass. `apply_ica` ignores `ica.exclude`; MNE's own `apply(exclude=...)`
  would remove the union of both.
- `method="infomax"` (extended Infomax) is the default and needs only MNE.
  `"picard"` (same model, faster) and `"fastica"` need
  `pip install eeg-decoder[ica]`. Fitting as many components as channels can
  be slow with Infomax (a 12-of-12 fit took ~3 min vs ~4 s for 10).
- **Z-score selection has a ceiling.** With *n* components no component can
  exceed a z-score of √(n−1), and MNE selects strictly above the threshold,
  so its default 3.0 never selects anything with ≤ 10 components.
  `find_artifact_components` raises in that case instead of silently finding
  nothing; use more components or `measure="correlation"` (e.g. 0.7). MNE
  also *iterates* the z-scoring, so a low threshold can add components whose
  correlation is near zero: check the reported correlations before applying.
- Order: mark bad channels (they are left out of the fit and untouched) ->
  ICA -> interpolate bad channels -> re-reference. After an average
  reference use at most (good EEG channels − 1) components.

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

## Signal quality

Descriptive metrics (never pass/fail verdicts) and plots, to compare
recordings and document data quality before and after preprocessing.

```python
import json
from pathlib import Path
from eeg_decoder.quality import (
    recording_quality, epochs_quality,
    plot_psd_comparison, plot_channel_quality, plot_rejection_summary,
)

q = recording_quality(raw, line_freq=50)     # line frequency is never assumed
print(q.summary())                           # bad channels, BAD time, amplitude, line noise
Path("quality.json").write_text(json.dumps(q.to_dict(), indent=2))  # machine-readable

eq = epochs_quality(epochs)                  # kept / dropped, and why (from drop_log)

fig = plot_psd_comparison(raw_original, raw_clean)   # mean PSD of good channels
fig.savefig("psd.png")                               # figures are returned, never shown or saved
```

- **`recording_quality`**: bad EEG channels and their fraction; time covered
  by `BAD*` annotations (union, clipped to the recording, exactly as epoching
  sees it) and per annotation description; per-channel robust SD and
  peak-to-peak in volts; and, with `line_freq`, a line-noise ratio: power
  within ±0.5 Hz of the line frequency vs the median power 2-5 Hz away, in
  dB (about 0 dB means no line noise, negative after a notch). Channel
  metrics use only data outside `BAD*` annotations.
- **`epochs_quality`**: events kept and dropped, drop reasons (channel names
  for threshold rejection, annotation descriptions, MNE's `NO_DATA` etc.),
  kept epochs per condition. Events MNE marks `IGNORED` (code not in
  `event_id`) are not counted as candidates. MNE does not keep the codes of
  dropped epochs, so per-condition counts cover kept epochs only.
- **Plots** use matplotlib's object-oriented `Figure`, not `pyplot`: no
  window, no global state, safe on headless servers. Colours come from a
  colour-vision-deficiency-checked palette, and bad channels are also
  hatched, so meaning never depends on colour alone.

## Optional feature extraction

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
| 3 | Artifact detection and correction (bad channels, optional ICA) | Done |
| 4 | Signal-quality assessment and visualization | Done |
| 5 | Optional fixed feature extraction (PSD, band power, Hjorth, time-frequency) | Done |
| 6 | EEG data augmentation (seeded, event/label preserving) | Next |
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
