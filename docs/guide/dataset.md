# Dataset preparation and export

Turn many recordings into one consistent dataset that any downstream tool
can load. Nothing here uses labels or is fitted to data.

```python
from eeg_decoder.dataset import prepare_recordings, save_npz, load_npz

recordings = [
    (raw_s01_t, {"subject": "01", "session": "T"}),
    (raw_s01_e, {"subject": "01", "session": "E"}),
    (raw_s02_t, {"subject": "02", "session": "T"}),
]
ds = prepare_recordings(recordings, config, ch_types=["eeg"])  # same config for each
print(ds.summary())

ds.X           # (n_epochs, n_channels, n_times), volts, read-only
ds.y           # event codes; ds.y_names gives condition names
ds.groups["subject"]   # one identifier per epoch, aligned after rejection
ds.records     # each recording's pipeline record

save_npz(ds, "dataset.npz")   # loads back exactly, with allow_pickle=False
ds = load_npz("dataset.npz")
```

- **Strict consistency.** Before combining, `check_compatible` compares
  every recording with the first: channel names and order, channel types and
  units, sampling rate, epoch start and length, filter edges, reference,
  baseline, bad channels (none allowed; interpolate or drop first), and the
  meaning of every event code. It reports **all** differences in one error;
  MNE's `concatenate_epochs` stops at the first. Nothing is resampled,
  re-referenced, interpolated, or reordered for you; `match_channels(inst,
  ch_names)` picks and reorders explicitly.
- **Failures.** By default the first failing recording stops the run, with
  a note naming it. With `on_error="collect"`, failing recordings (including
  ones with no epochs left) are skipped and listed in `ds.failures`;
  `ds.recording` keeps referring to positions in your input list.
- **Metadata and privacy.** Metadata values must be strings, numbers, or
  booleans, with the same keys for every recording; each key becomes a
  per-epoch array in `ds.groups` for downstream grouped splits. Only what you
  pass is stored: the measurement date and subject info in `raw.info` are
  never copied. pandas is not required; `ds.to_dataframe()` works if it is
  installed.
- **Channels.** `ch_types=["eeg"]` keeps only EEG channels in `X`; by default
  every channel (including stim) is kept, so nothing is dropped unless you ask.
- **Export.** `save_npz` writes plain arrays and JSON text only (no pickled
  objects) and refuses to overwrite unless `overwrite=True`.

## MNE `.fif` files per recording

```python
from eeg_decoder.dataset import save_epochs_fif, load_epochs_fif, combine_epochs

# while preparing: one <name>-epo.fif + <name>-epo.json per recording
ds = prepare_recordings(recordings, config, ch_types=["eeg"], fif_dir="prepared/")

# or for any epochs object
save_epochs_fif(epochs, "prepared/sub-01-epo.fif", metadata={"subject": "01"}, record=record)
epochs, sidecar = load_epochs_fif("prepared/sub-01-epo.fif")  # sidecar: metadata, record

# rebuild a dataset later from the files
loaded = [load_epochs_fif(p) for p in sorted(Path("prepared").glob("*-epo.fif"))]
ds = combine_epochs([e for e, _ in loaded], metadata=[s["metadata"] for _, s in loaded],
                    records=[s["record"] for _, s in loaded], ch_types=["eeg"])
```

- The `.fif` file keeps the full MNE `Epochs` (channel positions, filter
  settings, reference flag, baseline, events, `drop_log`); the JSON sidecar
  keeps what MNE does not: your metadata and the processing record.
- **Exact data.** MNE saves epochs in single precision by default, so data
  would reload only approximately; files here are written in double precision
  and round-trip exactly.
- **Anonymized by default.** A plain MNE save keeps the measurement date and
  subject information (name, birthday, hospital ID). The written copy is
  anonymized with MNE's `anonymize()` unless you pass `anonymize=False`; the
  sidecar records which was done.
- Names must end in `-epo.fif` (MNE's convention, which MNE itself does not
  enforce). Nothing is overwritten unless `overwrite=True`; with `fif_dir`,
  `prepare_recordings` refuses to start if a target file exists and writes
  files only after all recordings were combined successfully. With
  `fif_dir`, files hold every channel, independent of `ch_types`.
- The `.fif` format stores sampling rate, filter edges, epoch start, and
  baseline in single precision (a 0.1 Hz high-pass reloads as 0.1000000015),
  so the consistency check compares these with a small tolerance and reloaded
  epochs combine with freshly computed ones.
