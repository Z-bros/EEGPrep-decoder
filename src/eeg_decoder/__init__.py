"""EEG_Decoder: modular, reproducible EEG preprocessing and decoding.

Phase 1 exposes only the package version and the core configuration object.
"""

from eeg_decoder.config import EEGConfig

__version__ = "0.0.1"

__all__ = ["EEGConfig", "__version__"]
