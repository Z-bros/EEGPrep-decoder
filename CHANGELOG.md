# Changelog

All notable changes to EEGPrep-decoder (`eeg-decoder` on PyPI) are listed
here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Versioning

Versions are `MAJOR.MINOR.PATCH` ([Semantic Versioning](https://semver.org/)).
While the major version is 0, a **minor** release may change the public API;
every such change is listed under "Changed" or "Removed". Patch releases only
fix bugs and never change results on valid input except to correct an error,
which is stated here. Raising a minimum dependency version (e.g. MNE) is
always listed.

Every record written by the library (pipeline records, `.npz` exports, `.fif`
sidecars) stores the `eeg_decoder` version that produced it.

## [Unreleased]

## [0.1.0] - unreleased

First public release: a model-agnostic library that prepares EEG for any
downstream analysis and implements no models, training, or evaluation.

### Added

- **Input**: `io.to_raw` turns NumPy arrays into MNE objects with explicit
  units (converted to volts) and channel types.
- **Preprocessing**: `bandpass`, `notch`, `rereference`, `resample`, and a
  JSON-serializable `PreprocessingConfig` run by `EEGPipeline`, which returns
  a record of every step (parameters, versions, outputs).
- **Epoching**: `make_epochs`, `baseline_correct`.
- **Artifacts**: `assess_channels` (PREP-style criteria), `mark_bad_channels`,
  `interpolate_bad_channels`, `annotate_artifacts`, `add_annotations`,
  `reject_epochs`, and seeded, label-free ICA (`fit_ica`,
  `find_artifact_components`, `apply_ica`; extra `[ica]` for picard/fastica).
- **Signal quality**: `recording_quality`, `epochs_quality` and three plots.
- **Dataset preparation**: `check_compatible`, `match_channels`,
  `prepare_recordings`, `combine_epochs`, `EpochDataset`; export to `.npz`
  (no pickles) and to double-precision, anonymized `.fif` with JSON sidecars.
- **Optional features** (not used by the modules above): `welch_psd`,
  `band_power`, `time_stats`, `hjorth`, `morlet_tfr`, returned as named
  `FeatureArray`s.
- **Validation and docs**: real-data validation on PhysioNet EEGBCI, pinned
  MNE behaviours, a Sphinx site with executed example notebooks, and
  benchmarks against direct MNE calls (identical output, comparable speed).

### Requirements

Python >= 3.11; NumPy >= 2.1, SciPy >= 1.14, MNE-Python >= 1.13,
matplotlib >= 3.9, all tested at these minimum versions.

### Known limitations

FIR filters only; the pipeline holds about one extra copy of the data, and
`recording_quality` peaks at about five copies (see the benchmarks); no file
readers beyond MNE's.
