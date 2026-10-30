# EEG_Decoder

A modular, reproducible Python toolkit for EEG preprocessing and decoding.
Reusable functions instead of notebook code, built on the scientific Python
stack (NumPy, SciPy, MNE-Python, scikit-learn).

> **Status: pre-alpha (Phase 1 — package foundation).** Only the package
> skeleton, a core configuration object, and a synthetic-EEG test suite exist.
> Preprocessing, features, and decoding are developed incrementally and are
> not available yet.

## Installation

Requires Python 3.11 or newer.

```bash
git clone https://github.com/Z-bros/EEGPrep-decoder.git
cd EEGPrep-decoder
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Heavy dependencies (PyTorch, Braindecode, pyRiemann, MOABB) will be optional
extras so that preprocessing never requires the deep-learning stack.

## Quick check

```python
import eeg_decoder
from eeg_decoder import EEGConfig

print(eeg_decoder.__version__)
cfg = EEGConfig(seed=42)      # immutable; record it with every experiment
print(cfg.to_dict())          # {'seed': 42, 'float_dtype': 'float64'}
```

## Conventions (fixed from the start)

- Continuous EEG: `(n_channels, n_samples)`. Epoched EEG:
  `(n_epochs, n_channels, n_samples)`. Axes are never inferred silently.
- Signal units: volts (SI), as in MNE-Python, unless a function documents otherwise.
- Inputs are never modified in place unless a function explicitly says so.
- Every stochastic operation takes an explicit seed.

## Development

```bash
pip install -e ".[dev]"
pytest
```

No real EEG dataset is needed: tests use seeded synthetic signals with known
spectral content. Do not commit EEG recordings (see `.gitignore`).

## Known limitations

Pre-alpha. The API will change. Scientific assumptions for each operation
(filter type, referencing, units) will be documented as they are implemented.

## License

MIT — see [LICENSE](LICENSE).
