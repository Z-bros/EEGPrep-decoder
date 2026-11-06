# Benchmarks

How much do the library's checks and copies cost compared with calling
MNE-Python directly? [`benchmarks/run.py`](https://github.com/Z-bros/EEGPrep-decoder/blob/main/benchmarks/run.py)
times each wrapped operation next to the equivalent MNE call **with the same
semantics** (MNE on a `.copy()`, since the library never modifies its input),
checks that both give **identical output**, and measures peak memory.
Operations with no MNE equivalent are timed alone. Model accuracy is out of
scope.

## Method

- Synthetic 1/f EEG (~10 µV) with 50 Hz interference and a stim channel
  (events every 2 s), generated from a fixed seed.
- Sizes: `tiny` (8 ch, 250 Hz, 1 min; used by the tests), `small` (32 ch,
  250 Hz, 5 min), `medium` (64 ch, 500 Hz, 10 min; 149 MB per copy), and
  `large` (64 ch, 1000 Hz, 30 min; ~920 MB per copy).
- Time: median of 3 runs (wall clock). Memory: peak traced by `tracemalloc`
  in a separate run. It sees NumPy arrays but not scratch buffers inside
  compiled code (e.g. SciPy's FFT), so it is a lower bound.
- The full pipeline is notch -> band-pass -> average reference -> resample ->
  epochs -> baseline -> rejection, against the same chain written directly in
  MNE.

## Results (medium)

64 channels, 500 Hz, 10 min (149 MB per copy). Cloud workspace with 2 CPUs
(Intel Xeon, 2.1 GHz), Python 3.13, MNE 1.13.2. Absolute times depend on the
machine; the ratios and memory are the point.

| Operation | EEGPrep-decoder | MNE directly | Ratio | Identical output | Peak MB (ours / MNE) |
|---|---|---|---|---|---|
| bandpass 1-40 Hz | 0.506 s | 0.476 s | 1.06 | yes | 300 / 300 |
| notch 50 Hz | 0.534 s | 0.524 s | 1.02 | yes | 300 / 300 |
| rereference average | 0.171 s | 0.208 s | 0.82 | yes | 298 / 446 |
| resample to half rate | 0.951 s | 0.915 s | 1.04 | yes | 316 / 316 |
| make_epochs | 0.047 s | 0.046 s | 1.02 | yes | 75 / 75 |
| baseline_correct | 0.036 s | 0.033 s | 1.10 | yes | 74 / 74 |
| reject_epochs | 0.039 s | 0.059 s | 0.66 | yes | 74 / 74 |
| full pipeline (7 steps) | 2.152 s | 1.929 s | 1.12 | yes | 464 / 316 |
| assess_channels | 0.660 s | - | - | - | 311 / - |
| annotate_artifacts | 0.483 s | - | - | - | 440 / - |
| recording_quality | 1.237 s | - | - | - | 735 / - |
| band_power (features) | 0.334 s | - | - | - | 474 / - |

## Findings

- **The checks are cheap.** Each wrapped operation takes 0.66-1.12x MNE's
  time, and the pipeline 1.12x; differences of this size are run-to-run
  noise (ratios below 1 come from steps under 0.2 s). Validation
  (event checks, Nyquist and length checks, unit checks) does not touch the
  bulk of the data.
- **Outputs are identical.** Every wrapped operation, and the whole pipeline,
  returns exactly MNE's numbers (`np.array_equal`, not a tolerance). The
  library adds checks and records, never different arithmetic.
- **The pipeline holds one extra copy.** Each step copies its input, so the
  pipeline peaks at about 3 copies of the data (464 MB) where a hand-written
  MNE chain, filtering one copy in place, peaks at about 2 (316 MB). For a
  30-min, 64-channel, 1000 Hz recording that is roughly 2.8 vs 1.8 GB.
- **`recording_quality` is the most memory-hungry step**, about 5 copies of
  the data (735 MB here, ~4.5 GB at the large size), mostly from computing
  spectra and robust statistics over all channels at once.

The last two are candidates for optimisation in a later change (letting the
pipeline work on one private copy; computing quality metrics in channel
chunks). Both would keep outputs identical, which the benchmark checks.

## Running it

```bash
pip install -e .
python benchmarks/run.py --size small --out benchmark-report   # seconds
python benchmarks/run.py --size medium --out benchmark-report  # ~1-2 min, ~1 GB RAM
```

`large` needs several GB of RAM; run it on GitHub Actions instead: the
manual `benchmarks` workflow (Actions -> benchmarks -> Run workflow) runs any
size and attaches `benchmarks.json` / `benchmarks.md` to the run. The script
exits with 1 if any output differs from MNE's. `tests/test_benchmarks.py`
runs the `tiny` size on every test run, so a change that makes a wrapper
disagree with MNE fails the test suite.
