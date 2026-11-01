"""EEGPrep-decoder: model-agnostic, reproducible EEG preprocessing and data preparation.

The top-level namespace exposes the package version and the core
configuration object; functionality lives in the subpackages ``io``,
``preprocessing``, ``epoching``, ``artifacts``, and ``quality``.
"""

from eeg_decoder.config import EEGConfig

__version__ = "0.0.1"

__all__ = ["EEGConfig", "__version__"]
