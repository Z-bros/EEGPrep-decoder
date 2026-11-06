# MNE-Python behaviours that EEGPrep-decoder works around

EEGPrep-decoder builds on MNE-Python and keeps its behaviour wherever that is
safe. Where MNE silently drops data, only warns, or makes a choice the user
might not expect, this library raises, documents, or does the explicit thing
instead. Every such behaviour is listed here with the test that pins it.

All entries were verified on **MNE 1.13**. A test that pins *MNE's own*
behaviour fails when a new MNE release changes it; the workaround in the
middle column should then be revisited, and possibly removed.

"Wrapper test" means the test exercises our function, which passes the
behaviour through unchanged; it still fails if MNE changes.

## Filtering, referencing, resampling

| MNE behaviour | What EEGPrep-decoder does | Pinned by |
|---|---|---|
| A FIR filter longer than the signal only **warns** and returns distorted data. | `bandpass` / `notch` raise `ValueError`, stating the required length. | `test_mne_behaviour.py::test_fir_filter_longer_than_the_signal_only_warns` |
| `picks=None` filters EEG channels **including bad ones**, but not EOG, ECG, EMG, stim, or misc. | Kept and documented; `picks` is explicit when other types should be filtered. | `test_mne_behaviour.py::test_filter_picks_none_covers_eeg_including_bads_but_not_other_types` |
| `info["highpass"]` / `info["lowpass"]` change only when all data channels are filtered; a notch is never recorded in `info`. | Documented; the `EEGPipeline` record keeps every step and parameter. | `test_filtering.py::test_subset_picks_do_not_update_info_cutoffs` (wrapper test) |
| An average reference leaves bad channels **in the old reference**. | `rereference` refuses bad EEG channels unless `allow_bads=True`. | `test_mne_behaviour.py::test_average_reference_leaves_bad_channels_unreferenced` |
| Resampling **merges** stim events closer than one new sample; one silently disappears. | `resample` raises. | `test_mne_behaviour.py::test_resampling_merges_stim_events_closer_than_one_new_sample` |

## Epoching and rejection

| MNE behaviour | What EEGPrep-decoder does | Pinned by |
|---|---|---|
| Epochs whose window runs outside the data are **dropped silently** (`TOO_SHORT` / `NO_DATA` in `drop_log`). | `make_epochs` raises, listing the events. | `test_mne_behaviour.py::test_epochs_outside_the_data_are_dropped_silently` |
| A `decim` that aliases (new rate below 3x `info["lowpass"]`) only **warns**. | `make_epochs` raises. | `test_mne_behaviour.py::test_decimation_below_three_times_lowpass_only_warns` |
| `mne.Epochs` applies baseline `(None, 0)` **by default**. | `make_epochs` never applies a baseline; `baseline_correct` is a separate, explicit step. | `test_mne_behaviour.py::test_epochs_apply_a_baseline_by_default` |
| Dropping every epoch only **warns**. | `reject_epochs` raises, naming the channels responsible. | `test_mne_behaviour.py::test_dropping_every_epoch_only_warns` |
| `events_from_annotations` raises a generic error when no annotation matches, and returns nothing when there are no annotations. | The pipeline raises one explicit error listing the descriptions present. | `test_pipeline.py::test_annotations_without_matches` (wrapper test) |
| Without pandas, `epochs.metadata` is a **plain list of dicts** that cannot be queried. | Identifiers live in `EpochDataset.groups` (NumPy arrays); `to_dataframe()` when pandas is installed. | `test_mne_behaviour.py::test_epochs_metadata_without_pandas_is_a_plain_list_and_cannot_be_queried` (skipped where pandas is installed) |

## Artifacts and ICA

| MNE behaviour | What EEGPrep-decoder does | Pinned by |
|---|---|---|
| `annotate_amplitude` compares **consecutive samples**, so it misses smooth blinks. | `annotate_artifacts` uses a sliding-window peak-to-peak amplitude. | `test_annotations.py::test_mne_annotate_amplitude_misses_the_same_blink` |
| `interpolate_bads` silently **removes bad non-EEG channels** (e.g. EOG) from `bads`. | `interpolate_bad_channels` keeps non-EEG bad marks. | `test_channels.py::test_interpolation_keeps_bads_when_asked_and_keeps_non_eeg_bads` |
| ICA z-scores cannot exceed `sqrt(n - 1)` and selection is strictly greater, so the default threshold 3.0 **never selects anything** with 10 or fewer components. | `find_artifact_components` raises instead of returning nothing. | `test_ica.py::test_unreachable_zscore_threshold_raises_instead_of_finding_nothing` |
| ICA z-score selection is **iterated**, and can add components with near-zero correlation. | Documented; scores are returned for review; `measure="correlation"` avoids it. | Documented only (`test_ica.py::test_zscore_measure_with_reachable_threshold` tolerates it) |
| `ICA.apply(exclude=...)` removes the **union** with `ica.exclude`. | `apply_ica` removes exactly the listed components. | `test_mne_behaviour.py::test_ica_apply_removes_the_union_with_ica_exclude` |

## Saving and montages

| MNE behaviour | What EEGPrep-decoder does | Pinned by |
|---|---|---|
| `Epochs.save` writes **single precision** by default. | `save_epochs_fif` writes double precision; data round-trip exactly. | `test_fif.py::test_mne_default_single_precision_would_not_be_exact` |
| A plain save **keeps the measurement date and subject information**. | `save_epochs_fif` anonymizes by default. | `test_mne_behaviour.py::test_plain_fif_save_keeps_measurement_date_and_subject_info` |
| `.fif` stores sampling rate, filter edges, epoch start, and baseline as **float32**. | `check_compatible` compares them with a tolerance. | `test_fif.py::test_reloaded_epochs_are_still_compatible_with_fresh_ones` |
| The montage name `standard_1020` is **deprecated** (removal announced for MNE 1.14). | Docs and tests use `colin27_1020` / `colin27_1005`. | `test_mne_behaviour.py::test_standard_1020_montage_name_is_deprecated` |
