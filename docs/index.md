# EEGPrep-decoder

A reusable, **model-agnostic** Python library for EEG preprocessing and data
preparation, built on NumPy, SciPy, and MNE-Python. The import name is
`eeg_decoder`.

It turns raw recordings into clean, documented, analysis-ready data for any
downstream use (ERP analysis, statistics, machine learning, deep learning),
and stops there: it implements no models, training, or model evaluation, and
no supervised learned transforms such as CSP.

What makes it different from calling MNE directly:

- **Explicit, never silent.** Where MNE drops data silently or only warns
  (epochs outside the recording, aliasing decimation, filters longer than the
  signal, merged events), this library raises. See
  [MNE behaviours](mne-behaviours.md).
- **Units and metadata travel with the signal**: volts throughout, MNE
  objects in and out, inputs never modified.
- **Reproducible**: a JSON pipeline configuration and a record of every step,
  explicit seeds, and exact, anonymized exports.
- **Validated on real EEG**, with physiological checks
  ([scientific validation](validation.md)).

```{toctree}
:maxdepth: 1
:caption: Getting started

install
```

```{toctree}
:maxdepth: 2
:caption: User guide

guide/preprocessing
guide/artifacts
guide/quality
guide/dataset
guide/features
guide/reproducibility
```

```{toctree}
:maxdepth: 1
:caption: Examples

examples/01_raw_to_epochs
examples/02_many_recordings
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
module-notes
mne-behaviours
validation
benchmarks
ROADMAP
```
