"""Signal preprocessing operations. Each returns a new object; inputs are untouched."""

from eeg_decoder.preprocessing.filtering import bandpass, notch
from eeg_decoder.preprocessing.referencing import rereference
from eeg_decoder.preprocessing.resampling import resample

__all__ = ["bandpass", "notch", "rereference", "resample"]
