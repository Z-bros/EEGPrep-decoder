# Installation

Requires Python 3.11 or newer.

```bash
git clone https://github.com/Z-bros/EEGPrep-decoder.git
cd EEGPrep-decoder
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

The core install needs only NumPy, MNE-Python (which brings SciPy), and
matplotlib. Optional groups add only what a capability needs, so preparing
data never pulls in a modelling stack:

| Extra | Adds | For |
|---|---|---|
| `ica` | `python-picard`, `scikit-learn` | `fit_ica(method="picard")` or `"fastica"`; the default `"infomax"` needs neither |
| `dev` | `pytest` | running the test suite |
| `docs` | Sphinx, pydata theme, numpydoc, MyST-parser | building this documentation |

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
