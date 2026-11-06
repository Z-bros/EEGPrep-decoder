# Conventions and reproducibility

## Conventions (fixed from the start)

- Continuous EEG: `(n_channels, n_samples)`. Epoched EEG:
  `(n_epochs, n_channels, n_samples)`. Axes are never inferred silently.
- Signal units: volts (SI), as in MNE-Python, unless a function documents otherwise.
- Inputs are never modified in place unless a function explicitly says so.
- Every stochastic operation takes an explicit seed.

## Reproducibility

Save the `PreprocessingConfig` and the `record` from `EEGPipeline.run` with
your results (`json.dump(record, f, indent=2)`). Together they state what was
run, on what input, with which package versions, and what came out of each
step. No pipeline step is random, so the same config on the same data always
gives identical output; the only stochastic operation in the library, ICA,
requires an explicit seed.
