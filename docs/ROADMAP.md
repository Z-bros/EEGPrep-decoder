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
| 6 | Dataset preparation, metadata preservation, and export | Done (6a-6c) |
| 7 | Scientific validation, documentation, benchmarking, and release | Done; 0.1.0 awaits publishing |

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
| 6c | `save_epochs_fif`, `load_epochs_fif`; `prepare_recordings(..., fif_dir=...)` | Done |

Decisions:

- Identifiers live in the library's own per-epoch arrays (`groups`), so the
  core stays pandas-free; `EpochDataset.to_dataframe()` works when pandas is
  installed. Without pandas, MNE's `epochs.metadata` is only a plain list of
  dicts that cannot be queried (corrected in Phase 7a; pinned in
  `tests/test_mne_behaviour.py`).
- The first failing recording stops the run by default;
  `on_error="collect"` skips and lists failures instead.
- Every channel is kept unless `ch_types` (e.g. `["eeg"]`) is passed.
- Bad channels are not allowed when combining: interpolate or drop first.
- `.fif` files are written in double precision (MNE's default single
  precision does not round-trip exactly) and anonymized by default (a plain
  MNE save keeps the measurement date and subject information); both
  verified on MNE 1.13.
- `.fif` stores sampling rate, filter edges, epoch start, and baseline as
  float32, so `check_compatible` compares them with a tolerance (1e-6
  relative; half a sample for times) instead of exactly.

## In progress

### Phase 7: scientific validation, documentation, benchmarking, and release

| Sub-phase | Content | Status |
|---|---|---|
| 7a | Real-data validation on PhysioNet EEGBCI; MNE behaviours pinned and documented | Done, awaiting first real-data run |
| 7b | Documentation: Sphinx + numpydoc + pydata theme, user guide, API reference, notebooks, GitHub Pages | Done |
| 7c | Benchmarks of the library itself: runtime and peak memory per operation and pipeline, versus calling MNE directly | Done |
| 7d | Release `0.1.0` as `eeg-decoder`: changelog, CI on Python 3.11-3.13 x MNE 1.13 and latest, `build` + `twine check`, trusted publishing to PyPI on a version tag | Done; publishing is the owner's step |

**7a: scientific validation.**

- `validation/eegbci.py` (not part of the installed package) runs the library
  end to end on the EEG Motor Movement/Imagery dataset (Schalk et al. 2004;
  ODC-By 1.0), fetched with `mne.datasets.eegbci` (needs `pooch`): runs 1
  and 2 (eyes open / closed rest) and 3, 7, 11 (left/right fist movement).
- Pass/fail checks: units in volts; band-pass attenuation (>= 30 dB at
  55-75 Hz); notch lowers 60 Hz line noise; Berger effect (occipital relative
  alpha higher eyes-closed in >= 80 % of subjects); motor ERD (central
  8-30 Hz power lower in movement than rest in >= 70 %); seeded ICA exactly
  reproducible; every motor run passes `prepare_recordings`; exact `.npz`
  round trip; no subject errors. Thresholds are sanity criteria for robust
  effects; a sign-test p-value is reported, not used.
- Descriptive only: flagged bad channels, ICA component proposals (frontal
  channels as EOG proxy), epochs rejected.
- Runs as the manual / monthly `validation` GitHub workflow (the build
  workspace cannot reach PhysioNet); its logic is tested offline on synthetic
  EEG with known, and with reversed, effects.
- Finding: bad-channel assessment on an eyes-closed run flags occipital
  channels for their genuine alpha amplitude (reproduced on synthetic data),
  and interpolating them would erase the effect. The script assesses the
  eyes-open run and applies its bad channels to both runs.
- [docs/mne-behaviours.md](mne-behaviours.md) lists 20 MNE 1.13 behaviours
  the library works around, each with its pinning test; 12 new direct pins
  are in `tests/test_mne_behaviour.py`. Writing them corrected one earlier
  claim (MNE metadata without pandas, Phase 6).

**7b: documentation.**

- Sphinx site in `docs/` (pydata theme, numpydoc, MyST Markdown): landing
  page, installation, a user guide per area (moved from the README, which is
  now a short overview), an API reference generated from the docstrings, the
  module docstrings as "scientific notes by module", the MNE behaviours, the
  validation, and this roadmap. Guide pages stay plain Markdown so they also
  read well on GitHub.
- The `docs` workflow builds with warnings as errors on every push and pull
  request; it deploys to GitHub Pages from `main` once Pages is enabled
  (Source: GitHub Actions) and the repository variable `DOCS_DEPLOY` is
  `true`. Install with the `docs` extra.
- The strict build caught four docstrings whose `|x|` notation was read as
  a reStructuredText substitution; "see module notes" became links.
- Two example notebooks in `docs/examples/` (synthetic data, no downloads):
  one recording from raw to clean epochs (quality, bad channels, a saved
  configuration, PSD and rejection plots, an ERP, optional features), and
  many recordings into one dataset (consistency errors, collecting failures,
  `.npz` / `.fif` export, leave-one-subject-out indices). They are stored
  without outputs (a test enforces it) and executed by MyST-NB on every docs
  build, so a broken example fails CI.

**7c: benchmarks.**

- `benchmarks/run.py` (not part of the installed package) times every
  wrapped operation and the full pipeline next to the same MNE calls, checks
  that outputs are identical, and measures peak memory (`tracemalloc`, a
  lower bound), on seeded synthetic data at four sizes up to 64 ch x 1000 Hz
  x 30 min. Results and method: [benchmarks](benchmarks.md).
- `tests/test_benchmarks.py` runs the smallest size in the test suite, so a
  wrapper that stops matching MNE fails CI; the manual `benchmarks` workflow
  runs any size (including `large`) on GitHub's runners.
- Findings (medium size, 149 MB per copy): time 0.66-1.12x MNE (noise),
  outputs identical; the pipeline holds about one extra copy of the data
  (464 vs 316 MB) because each step copies; `recording_quality` peaks at
  about 5 copies. Both are optimisation candidates for a later change, with
  the benchmark guarding that outputs stay identical.

**7d: release preparation.**

- Version `0.1.0`, `CHANGELOG.md` (Keep a Changelog) with the versioning
  policy: semantic versioning; while 0.x, minor releases may change the API
  and every change is listed.
- Minimum versions are tested, not assumed: NumPy 2.1, SciPy 1.14 (now a
  declared dependency, since the library imports it directly), MNE 1.13, and
  matplotlib 3.9 on Python 3.11 pass the full suite. The `minimum` CI job
  installs them from `.github/minimum-versions.txt`; a test keeps that file
  equal to the floors in `pyproject.toml`.
- The `package` CI job builds the sdist and wheel, runs `twine check
  --strict`, and runs the test suite against the installed wheel. The sdist
  includes `docs/`, `validation/`, and `benchmarks/`, so its tests run too.
- `release.yml` publishes to PyPI on a `v*.*.*` tag through trusted
  publishing (no stored token), after checking that the tag, `__version__`,
  and a dated changelog entry agree; a manual run publishes to TestPyPI.
- README links are absolute so they work on PyPI; project URLs, author, and
  the Alpha classifier are set.
- Publishing needs the owner: register the trusted publisher on PyPI, date
  the changelog entry, and push a signed tag.

## Changes from the original roadmap

The original plan included classical decoding (Phase 4), deep learning
(Phase 5), and fine-tuning / knowledge distillation (Phase 6), with CSP as a
feature. These are removed from the library's scope. The import name
`eeg_decoder` and the distribution name `eeg-decoder` are unchanged to keep
the existing public API stable.
