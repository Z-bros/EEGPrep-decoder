# Artifact handling

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
- **Assess on data without a strong, focal rhythm.** A large physiological
  rhythm makes channels amplitude outliers: with eyes closed, occipital alpha
  can get O1/Oz/O2 flagged, and interpolating them erases the alpha itself.
  Electrode faults belong to the cap, so assess a condition such as eyes-open
  rest and apply those bad channels to the other runs of the session (found
  in the [scientific validation](../validation.md)).
- All of these work per recording without labels, so they cause no leakage.

## Optional ICA (blinks, eye movements, heartbeat)

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
