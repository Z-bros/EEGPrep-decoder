"""Signal preprocessing operations. Each returns a new object; inputs are untouched."""

from eeg_decoder.preprocessing.filtering import bandpass, notch

__all__ = ["bandpass", "notch"]
