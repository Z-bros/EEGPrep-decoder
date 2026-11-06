# EEGPrep-decoder

A reusable, **model-agnostic** Python library for EEG preprocessing and data
preparation, inspired by Med_Im_Prepper. Reusable, tested functions instead
of notebook code, built on NumPy, SciPy, and MNE-Python. The import name is
`eeg_decoder`.

It turns raw recordings into clean, documented, analysis-ready data for any
downstream use (ERP analysis, statistics, machine learning, deep learning),
and stops there.

> **Status: alpha (0.1.0).** Preprocessing and a reproducible pipeline,
> artifact handling (bad channels, annotations, optional ICA), signal-quality
> reports and plots, optional features, and dataset preparation and export,
> validated on real EEG and benchmarked against MNE. While the version is
> 0.x, minor releases may change the API; every change is in the
> [changelog](https://github.com/Z-bros/EEGPrep-decoder/blob/main/CHANGELOG.md). See the [roadmap](https://github.com/Z-bros/EEGPrep-decoder/blob/main/docs/ROADMAP.md).

## Scope

**In scope:** preprocessing, artifact detection and correction (including
bad-channel assessment and optional ICA), signal-quality assessment and
visualization, *optional* fixed feature extraction (PSD, band power, Hjorth,
time-frequency), dataset preparation with metadata preservation, export of
prepared arrays or MNE objects, and scientific validation of the library
itself.

**Out of scope:** classification models, model training, fine-tuning,
knowledge distillation, and model evaluation frameworks. **Supervised
learned transformations such as CSP are not part of the library**; they
belong in your downstream modelling code (e.g. a scikit-learn `Pipeline`
with `mne.decoding.CSP`), fitted on training data only.

Feature extraction is optional and independent: it is not an `EEGPipeline`
step, and preparing EEG data never requires it.

## Installation

Requires Python 3.11 or newer.

```bash
pip install eeg-decoder           # once 0.1.0 is on PyPI; extras: "eeg-decoder[ica]"
```

From source (for development):

```bash
git clone https://github.com/Z-bros/EEGPrep-decoder.git
cd EEGPrep-decoder
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"           # extras: ".[ica]", ".[docs]"
```

The core install needs only NumPy, SciPy, MNE-Python, and matplotlib, each
tested at its minimum supported version; see [Installation](https://github.com/Z-bros/EEGPrep-decoder/blob/main/docs/install.md)
for the optional groups.

## Quick start

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

Epoching, baseline correction, and rejection, for a recording with a stim
channel (or use annotations):

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

The same chain runs from a saved JSON configuration, with a record of every
step, through `EEGPipeline`; see the [preprocessing guide](https://github.com/Z-bros/EEGPrep-decoder/blob/main/docs/guide/preprocessing.md).

## Documentation

The documentation is a Sphinx site built from [`docs/`](https://github.com/Z-bros/EEGPrep-decoder/tree/main/docs); its
pages are Markdown and also read well on GitHub.

- User guide: [preprocessing and pipelines](https://github.com/Z-bros/EEGPrep-decoder/blob/main/docs/guide/preprocessing.md),
  [artifact handling](https://github.com/Z-bros/EEGPrep-decoder/blob/main/docs/guide/artifacts.md),
  [signal quality](https://github.com/Z-bros/EEGPrep-decoder/blob/main/docs/guide/quality.md),
  [dataset preparation and export](https://github.com/Z-bros/EEGPrep-decoder/blob/main/docs/guide/dataset.md),
  [optional features](https://github.com/Z-bros/EEGPrep-decoder/blob/main/docs/guide/features.md),
  [conventions and reproducibility](https://github.com/Z-bros/EEGPrep-decoder/blob/main/docs/guide/reproducibility.md)
- Reference: the API reference (generated from the docstrings), scientific
  notes by module, and [MNE behaviours](https://github.com/Z-bros/EEGPrep-decoder/blob/main/docs/mne-behaviours.md) the library
  works around, each pinned by a test
- Example notebooks, run on every docs build:
  [from a raw recording to clean epochs](https://github.com/Z-bros/EEGPrep-decoder/blob/main/docs/examples/01_raw_to_epochs.ipynb),
  [many recordings, one dataset](https://github.com/Z-bros/EEGPrep-decoder/blob/main/docs/examples/02_many_recordings.ipynb)
- [Scientific validation](https://github.com/Z-bros/EEGPrep-decoder/blob/main/docs/validation.md) on real EEG (PhysioNet EEGBCI)
- [Benchmarks](https://github.com/Z-bros/EEGPrep-decoder/blob/main/docs/benchmarks.md): runtime and memory versus calling MNE
  directly (same speed, identical output)

Build it locally:

```bash
pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs docs/_build/html
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

No real EEG dataset is needed: tests use seeded synthetic signals with known
spectral content. Do not commit EEG recordings (see `.gitignore`).

## Roadmap

Full details, design rules, and per-phase considerations:
[docs/ROADMAP.md](https://github.com/Z-bros/EEGPrep-decoder/blob/main/docs/ROADMAP.md).

| Phase | Topic | Status |
|---|---|---|
| 1 | Package foundation | Done |
| 2 | Core preprocessing and `EEGPipeline` | Done |
| 3 | Artifact detection and correction (bad channels, optional ICA) | Done |
| 4 | Signal-quality assessment and visualization | Done |
| 5 | Optional fixed feature extraction (PSD, band power, Hjorth, time-frequency) | Done |
| 6 | Dataset preparation, metadata preservation, and export | Done |
| 7 | Scientific validation, documentation, benchmarking, and release | Done; 0.1.0 awaits publishing |

## Known limitations

Alpha: while the version is 0.x, minor releases may change the API (listed
in the [changelog](https://github.com/Z-bros/EEGPrep-decoder/blob/main/CHANGELOG.md)).

- Only FIR filtering is exposed; IIR filters and custom transition bandwidths
  are not yet configurable.
- When the functions are called directly, MNE's `raw.info` records band-pass
  cutoffs but not notch filtering, and only *that* the reference changed.
  Use `EEGPipeline`, whose record keeps every step's parameters.
- No REST reference, and no restoring of an unrecorded online reference
  electrode (`mne.add_reference_channels`).
- No file readers yet: load recordings with MNE (e.g. `mne.io.read_raw_edf`)
  and pass the resulting `Raw` object in directly.

## License

MIT — see [LICENSE](https://github.com/Z-bros/EEGPrep-decoder/blob/main/LICENSE).
