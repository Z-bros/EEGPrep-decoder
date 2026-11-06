# Installation

Requires Python 3.11 or newer.

```bash
pip install eeg-decoder          # once 0.1.0 is on PyPI
pip install "eeg-decoder[ica]"   # with an optional group
```

From source:

```bash
git clone https://github.com/Z-bros/EEGPrep-decoder.git
cd EEGPrep-decoder
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

The core install needs only NumPy, SciPy, MNE-Python, and matplotlib, each
tested in CI at its minimum supported version (NumPy 2.1, SciPy 1.14, MNE
1.13, matplotlib 3.9, on Python 3.11). Optional groups add only what a
capability needs, so preparing data never pulls in a modelling stack:

| Extra | Adds | For |
|---|---|---|
| `ica` | `python-picard`, `scikit-learn` | `fit_ica(method="picard")` or `"fastica"`; the default `"infomax"` needs neither |
| `dev` | `pytest` | running the test suite |
| `docs` | Sphinx, pydata theme, numpydoc, MyST-parser, MyST-NB, ipykernel | building this documentation and running its notebooks |

```bash
pip install -e ".[ica]"     # or ".[dev]", ".[docs]", ".[dev,ica]"
```

## Quick check

```python
import eeg_decoder
from eeg_decoder import EEGConfig

print(eeg_decoder.__version__)
cfg = EEGConfig(seed=42)      # immutable; record it with every experiment
print(cfg.to_dict())          # {'seed': 42, 'float_dtype': 'float64'}
```

A bare `import eeg_decoder` loads neither MNE nor matplotlib; each subpackage
imports what it needs.

## Development

```bash
pip install -e ".[dev]"
pytest
```

No real EEG dataset is needed: tests use seeded synthetic signals with known
spectral content. Do not commit EEG recordings (see `.gitignore`).

Build the documentation (warnings are errors, as in CI):

```bash
pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs docs/_build/html
```
