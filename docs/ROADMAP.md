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
- Scientific validation, documentation, and benchmarking of the library
  itself.

### Out of scope

The library does **not** implement, now or in later phases:

- EEG classification or regression models (classical or deep learning);
- model training, fine-tuning, or knowledge distillation;
- model evaluation frameworks (cross-validation, metrics, benchmarks of
  model accuracy);
- **supervised learned transformations** such as CSP, xDAWN, or any spatial
  filter or scaler fitted using labels.

Supervised transforms stay downstream, inside the user's own modelling code
(e.g. a scikit-learn `Pipeline` with `mne.decoding.CSP`), where they can be
fitted on training data only. The documentation may show how prepared data
feeds into such code, but that code is not part of the library.

## Design rules that apply to every phase

1. **The preprocessing pipeline never requires feature extraction.**
   Features live in a separate, optional subpackage, are not steps of
   `EEGPipeline`, and are never imported by the preprocessing, epoching,
   artifact, or quality modules. Preparing EEG data never depends on them.
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
| 4 | Signal-quality assessment and visualization | Done (4a-4b) |
| 5 | Optional fixed feature extraction | Done |
| 6 | Dataset preparation, metadata preservation, and export | In progress (6a, 6b, `.npz` done) |
| 7 | Scientific validation, documentation, benchmarking, and release | Planned |

Each phase starts with a design review before code is written and is split
into small sub-milestones. Function names in planned phases are proposals.

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

### Phase 3: artifact detection and correction

Find and handle bad channels and artifacts before epoching, so that
re-referencing and epoch rejection work on trustworthy data.

| Sub-phase | Public API |
|---|---|
| 3a | `artifacts.assess_channels`, `ChannelAssessment`, `mark_bad_channels`, `interpolate_bad_channels` (also the pipeline op `interpolate_bad_channels`) |
| 3b | `artifacts.annotate_artifacts`, `add_annotations` |
| 3c | `artifacts.fit_ica`, `find_artifact_components`, `ComponentReport`, `apply_ica`; `[ica]` extra for picard/fastica |

- Bad-channel assessment (per recording, no labels) reports *why* each
  channel was flagged; marking channels bad is an explicit, separate step,
  and assessment is deliberately not a pipeline op so a person reviews it.
- Interpolation refuses missing positions and too many bad channels, and
  keeps the bad marks of non-EEG channels.
- Artifact annotations use a sliding-window peak-to-peak amplitude.
- ICA is seeded and label-free; fit, component proposal, and removal are
  separate steps, and only the components passed are removed. `infomax`
  works with MNE alone.

### Phase 4: signal-quality assessment and visualization

| Sub-phase | Public API |
|---|---|
| 4a | `quality.recording_quality` / `RecordingQuality`, `quality.epochs_quality` / `EpochsQuality` |
| 4b | `quality.plot_psd_comparison`, `plot_channel_quality`, `plot_rejection_summary` |

Metrics are descriptive, never pass/fail verdicts, and JSON-serializable.
Plots return matplotlib figures and never show, block, or save. Not done: a
channel-quality *topography* (needs electrode positions); the per-channel bar
chart covers the same information without a montage.

### Phase 5: optional fixed feature extraction

| Public API (`eeg_decoder.features`) |
|---|
| `FeatureArray` (named axes, units, aligned `events`; `flatten()`) |
| `welch_psd`, `band_power`, `CLASSIC_BANDS` |
| `time_stats`, `hjorth` |
| `morlet_tfr` |

Entirely optional: a separate subpackage, not an `EEGPipeline` op, and never
imported by the preprocessing modules. Every feature is computed from a
single epoch's own samples, so it is leakage-free. Window lengths and bands
are always explicit. Input is `mne.Epochs` (arrays via `mne.EpochsArray`).

## Planned

### Phase 6: dataset preparation, metadata preservation, and export

Goal: turn many recordings into one consistent, documented dataset that any
downstream tool can load, without any modelling in this library.

- **6a: consistency across recordings.** Compare channel set and order,
  sampling rate, `event_id`, epoch window and length, filter edges,
  reference, and units, and report *all* differences in one explicit error
  (MNE's `concatenate_epochs` stops at the first). A helper picks and
  reorders channels to an explicit list; nothing is silently interpolated,
  resampled, re-referenced, or dropped.
- **6b: running many recordings.** Apply a `PreprocessingConfig` to each
  `(raw, metadata)` pair (e.g. subject, session, run, dataset), keep every
  pipeline record, and combine the results into one dataset with `X`
  `(n_epochs, n_channels, n_times)` in volts, `y` and `event_id`, one
  identifier per epoch for each metadata key (so downstream grouped splits by
  subject or session are possible), channel names, sampling rate, times, and
  the records. Stopping at the first failing recording is the proposed
  default, with an option to collect failures instead.
- **6c: export.** NumPy archives (`.npz`) that round-trip everything and load
  with `allow_pickle=False`, and MNE `.fif` files per recording with a JSON
  sidecar (record and metadata). Further formats (e.g. BIDS derivatives via
  `mne-bids`) only behind optional extras.

Metadata preservation and privacy: only metadata the user passes is stored;
the measurement date and subject information inside `raw.info` are never
copied. Identifiers are kept aligned with the epochs after rejection.

| Sub-phase | Public API (`eeg_decoder.dataset`) | Status |
|---|---|---|
| 6a | `check_compatible`, `match_channels` | Done |
| 6b | `prepare_recordings`, `combine_epochs`, `EpochDataset` | Done |
| 6c | `save_npz`, `load_npz` | Done |
| 6c | `.fif` export per recording with JSON sidecar | Next |

Decisions:

- Identifiers live in the library's own per-epoch arrays (`groups`), so the
  core stays pandas-free; `EpochDataset.to_dataframe()` works when pandas is
  installed. MNE's `epochs.metadata` would have required pandas (verified on
  MNE 1.13).
- The first failing recording stops the run by default;
  `on_error="collect"` skips and lists failures instead.
- Every channel is kept unless `ch_types` (e.g. `["eeg"]`) is passed.
- Bad channels are not allowed when combining: interpolate or drop first.

### Phase 7: scientific validation, documentation, benchmarking, and release

- **Scientific validation.** Check every operation against trusted reference
  implementations (MNE-Python, SciPy) and analytic expectations, building on
  the existing tests; run the full workflow on public EEG datasets (licensing
  checked) and document the outcomes, e.g. that filters, referencing,
  bad-channel assessment, and ICA behave as described; record every MNE
  behaviour the library works around, with a test pinning it, so a change in
  MNE is caught.
- **Documentation.** User guide and API reference (e.g. Sphinx or MkDocs)
  stating the scientific assumptions of every operation, plus example
  notebooks; reusable code stays in the package.
- **Benchmarking of the library itself.** Runtime and memory of
  preprocessing steps and pipelines, and numerical agreement with reference
  implementations. No model-accuracy benchmarks.
- **Release.** Versioning policy, changelog, CI across supported Python and
  MNE versions, packaging checks, and a first PyPI release.

## Changes from the original roadmap

The original plan included classical decoding (Phase 4), deep learning
(Phase 5), and fine-tuning / knowledge distillation (Phase 6), with CSP as a
feature. These are removed from the library's scope. The import name
`eeg_decoder` and the distribution name `eeg-decoder` are unchanged to keep
the existing public API stable.
