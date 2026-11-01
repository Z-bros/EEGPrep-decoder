# EEGPrep-decoder: scope and roadmap

EEGPrep-decoder (import name `eeg_decoder`) is a reusable, **model-agnostic**
EEG preprocessing and data-preparation library, built in the spirit of
Med_Im_Prepper (`medprep-starter`): small, reusable, well-tested functions
that turn raw recordings into clean, documented, analysis-ready data.

It prepares data *for* any downstream analysis (classical statistics, ERP
analysis, machine learning, deep learning) without implementing that
analysis itself.

## Scope

### In scope

- Converting arrays to MNE objects with explicit units and metadata.
- Signal preprocessing: filtering, re-referencing, resampling, epoching,
  baseline correction, artifact rejection, and a serializable pipeline.
- Artifact detection and correction, including bad-channel assessment and
  optional ICA.
- Signal-quality assessment and visualization.
- **Optional**, fixed (non-learned) feature extraction: PSD, band power,
  Hjorth parameters, time-frequency representations.
- Dataset preparation, metadata preservation, and export of prepared arrays
  or MNE objects.

### Out of scope

The library does **not** implement, now or in later phases:

- EEG classification or regression models (classical or deep learning);
- model training, fine-tuning, or knowledge distillation;
- model evaluation frameworks (cross-validation, metrics, benchmarks of
  model accuracy);
- **supervised learned transformations** such as CSP, xDAWN, or any spatial
  filter or scaler fitted using labels;
- data augmentation (removed from the roadmap after Phase 5).

Supervised transforms stay downstream, inside the user's own modelling code
(e.g. a scikit-learn `Pipeline` with `mne.decoding.CSP`), where they can be
fitted on training data only. The documentation may show how prepared data
feeds into such code, but that code is not part of the library.

## Design rules that apply to every phase

1. **The preprocessing pipeline never requires feature extraction.**
   Features live in a separate, optional subpackage and are not steps of
   `EEGPipeline`. Preparing EEG data never depends on them.
2. **No step uses labels.** Anything in the library that is estimated from
   data (ICA unmixing, robust bad-channel statistics, quality metrics) is
   unsupervised and computed per recording. Component or channel selection
   must not look at class labels.
3. **Leakage is documented, not hidden.** Where an operation is fitted to
   data, its docs state what it was fitted on, and offer a way to fit on one
   set of data and apply to another.
4. **Reproducibility.** Every stochastic operation (e.g. ICA) takes an
   explicit seed, and records keep parameters, versions, and seeds.
5. **MNE-first, copy-not-mutate, explicit units (volts)**, as in Phases 1-2.
6. **Optional dependency groups** keep the core install small; each phase
   adds an extra only for what it needs.

## Status

| Phase | Topic | Status |
|---|---|---|
| 1 | Package foundation | Done |
| 2 | Core preprocessing and `EEGPipeline` | Done (2a-2d) |
| 3 | Artifact detection and correction | Done (3a-3c) |
| 4 | Signal-quality assessment and visualization | Done |
| 5 | Optional fixed feature extraction | Done |
| 6 | Dataset preparation, metadata, and export | Next |
| 7 | Documentation, benchmarking, and release | Planned |

Function names below are proposals; each phase starts with a design review
before any code is written, and is split into small sub-milestones.

## Completed

### Phase 1: package foundation

`pyproject.toml` (src layout, Python >= 3.11), `EEGConfig`, MIT license, CI
workflow, seeded synthetic EEG fixtures.

### Phase 2: core preprocessing

| Sub-phase | Public API |
|---|---|
| 2a | `io.to_raw`, `preprocessing.bandpass`, `preprocessing.notch` |
| 2b | `preprocessing.rereference`, `preprocessing.resample` |
| 2c | `epoching.make_epochs`, `epoching.baseline_correct`, `artifacts.reject_epochs` |
| 2d | `preprocessing.PreprocessingConfig`, `preprocessing.EEGPipeline` |

These APIs and their tests are kept unchanged by the roadmap revision.

## Planned

### Phase 3: artifact detection and correction

Goal: find and handle bad channels and artifacts before epoching, so that
re-referencing and epoch rejection work on trustworthy data.

| Sub-phase | Public API | Status |
|---|---|---|
| 3a | `artifacts.assess_channels`, `ChannelAssessment`, `mark_bad_channels`, `interpolate_bad_channels` (+ pipeline op) | Done |
| 3b | `artifacts.annotate_artifacts`, `add_annotations` | Done |
| 3c | `artifacts.fit_ica`, `find_artifact_components`, `ComponentReport`, `apply_ica`; `[ica]` extra for picard/fastica | Done |

- **Bad-channel assessment** (per recording, no labels): flat or
  near-constant channels, extreme variance or amplitude (robust z-scores),
  low correlation with neighbouring channels, excessive line noise or
  high-frequency power. Returns a report of *why* each channel was flagged;
  marking channels bad is an explicit, separate step.
- **Interpolation** of bad channels via MNE (requires electrode positions,
  so a montage helper is part of this phase).
- **Annotation helpers**: mark segments (e.g. amplitude bursts, flat
  stretches) as `BAD_*` annotations, which `make_epochs` already honours.
- **Optional ICA** via `mne.preprocessing.ICA`: seeded fitting (with a
  recommended 1 Hz high-passed copy for fitting), EOG/ECG component
  detection with MNE's correlation methods, explicit component exclusion,
  and applying the solution to the original data. Fit and apply are separate
  so a solution can be reused or audited.
- Pipeline integration: label-free steps (e.g. interpolation) may become
  `EEGPipeline` ops; detection reports stay outside the pipeline so a human
  can review them.

Considerations: robust statistics on short recordings, the order of
operations (bad channels before average reference), and seeding. MNE's
default `fastica` method needs scikit-learn and `picard` needs
`python-picard`; only `infomax` works with MNE alone, so ICA ships as an
optional extra (e.g. `pip install eeg-decoder[ica]`).

### Phase 4: signal-quality assessment and visualization

Goal: make data quality visible and quantitative, before and after
preprocessing.

- Per-channel and per-recording quality metrics: fraction of bad channels,
  fraction of data marked bad, line-noise ratio, amplitude and variance
  distributions, epochs rejected and why (from `drop_log` and the pipeline
  record).
- A machine-readable quality report (JSON-serializable, like the pipeline
  record).
- Plots built on MNE/matplotlib: PSD before vs after preprocessing,
  channel-quality topography, rejection summary. Plotting functions return
  figures and never block or save implicitly.

Considerations: metrics are descriptive, not pass/fail verdicts; thresholds
are user-supplied. Plot tests use a non-interactive backend.

| Sub-phase | Public API | Status |
|---|---|---|
| 4a | `quality.recording_quality` / `RecordingQuality`, `quality.epochs_quality` / `EpochsQuality` | Done |
| 4b | `quality.plot_psd_comparison`, `plot_channel_quality`, `plot_rejection_summary` | Done |

Deferred: a channel-quality *topography* (needs electrode positions); the
per-channel bar chart covers the same information without a montage.
Summaries of the `EEGPipeline` record are not separate functions: the record
is already JSON, and `epochs_quality` covers the rejection part.

### Phase 5: optional fixed feature extraction

Goal: provide common, non-learned EEG descriptors for users who want them.
Entirely optional: separate `features` subpackage, not an `EEGPipeline` op.

- Welch PSD with explicit window length (frequency resolution) and window
  type; parity with SciPy and MNE.
- Absolute and relative band power with **user-supplied** bands (a
  documented example band set is offered, never applied implicitly);
  half-open `[low, high)` bands; validation against Nyquist and resolution.
- Hjorth activity, mobility, and complexity (with guidance to low-pass
  first, since mobility and complexity are very noise-sensitive).
- Time-frequency representations (e.g. Morlet wavelets via MNE) with
  explicit frequencies and cycles.
- Outputs: arrays with explicit axes plus interpretable feature names
  (e.g. `"C3:alpha"`), aligned row-for-row with `epochs.events`.

Every feature is computed from a single epoch's own samples, so it is
leakage-free. Supervised transforms such as CSP are explicitly excluded (see
Out of scope).

| Public API (`eeg_decoder.features`) | Status |
|---|---|
| `FeatureArray` (named axes, units, aligned `events`; `flatten()`) | Done |
| `welch_psd`, `band_power`, `CLASSIC_BANDS` | Done |
| `time_stats`, `hjorth` | Done |
| `morlet_tfr` | Done |

Decisions: input is `mne.Epochs` only (arrays via `mne.EpochsArray`); no
array adapter was added. Features are not `EEGPipeline` ops, and importing
the preprocessing, epoching, artifact, or quality modules never imports them.

### Phase 6: dataset preparation, metadata, and export

Goal: turn many recordings into a consistent, documented dataset.

- Running a `PreprocessingConfig` over multiple recordings with per-recording
  metadata (subject, session, run, dataset), and collecting the records.
- Harmonization checks across recordings: same channels and order, same
  sampling rate, same `event_id`, same units; explicit errors when they
  differ, plus helpers to pick or reorder channels.
- Metadata preservation: subject/session/run identifiers carried into
  `epochs.metadata` so downstream grouped splits (by subject or session) are
  possible; privacy guidance on identifying fields.
- Export: MNE `.fif` files, and NumPy archives (`.npz`) with `X`
  `(n_epochs, n_channels, n_times)`, `y`, `groups`, channel names, sampling
  rate, and the processing record. Optional formats (e.g. BIDS derivatives
  via `mne-bids`, HDF5) behind optional extras.

### Phase 7: documentation, benchmarking, and release

- User guide and API reference (e.g. Sphinx or MkDocs), with the scientific
  assumptions of every operation, and example notebooks on public datasets
  (licensing checked).
- Benchmarks of the **library itself**: runtime and memory of preprocessing
  steps and pipelines, and numerical agreement with reference
  implementations. No model-accuracy benchmarks.
- Release preparation: versioning policy, changelog, CI across supported
  Python and MNE versions, packaging checks, and a first PyPI release.

## Changes from the original roadmap

The original plan included classical decoding (Phase 4), deep learning
(Phase 5), and fine-tuning / knowledge distillation (Phase 6), with CSP as a
feature. These are removed from the library's scope. The import name
`eeg_decoder` and the distribution name `eeg-decoder` are unchanged to keep
the existing public API stable.

After Phase 5, EEG data augmentation was removed from the roadmap entirely
(it had been planned as Phase 6). Dataset preparation and export became
Phase 6, and documentation and release Phase 7.
