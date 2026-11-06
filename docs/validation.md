# Scientific validation

Beyond unit tests, the library is checked on real EEG:
[`validation/eegbci.py`](https://github.com/Z-bros/EEGPrep-decoder/blob/main/validation/eegbci.py) runs it end to end on the
PhysioNet EEG Motor Movement/Imagery dataset (EEGBCI; Schalk et al. 2004,
ODC-By 1.0) and checks known physiology, not just that the code runs:

- a 1-40 Hz band-pass removes >= 30 dB of 55-75 Hz power, and a 60 Hz notch
  lowers line noise, on every subject;
- **Berger effect**: occipital alpha is higher with eyes closed than open;
- **motor ERD**: central 8-30 Hz power drops during fist movement;
- ICA with a fixed seed is exactly reproducible, and the multi-recording
  dataset round-trips through `.npz` exactly.

```bash
pip install -e . pooch          # pooch downloads the data via MNE
python validation/eegbci.py --subjects 1-10 --out validation-report
```

It writes `report.json`, `summary.md`, and figures, and exits with 1 if a
check fails. On GitHub it runs as the manual / monthly `validation` workflow,
whose report is attached to the run. Its logic is tested offline on
synthetic EEG with known effects (`tests/test_validation_eegbci.py`).

One lesson it encoded: run bad-channel assessment on data without a strong,
focal rhythm. With eyes closed, occipital alpha can make O1/Oz/O2 amplitude
outliers, and interpolating them would erase the alpha itself. The script
assesses the eyes-open run and applies those bad channels to both runs.

Every MNE behaviour the library works around (silent drops, warnings it
turns into errors, defaults it avoids) is listed with the test that pins it in
[MNE behaviours](mne-behaviours.md), so an MNE upgrade that
changes one is caught.
