"""Optional, fixed (non-learned) EEG features, computed per epoch.

Nothing here is needed to prepare EEG data, and nothing here is a step of
``EEGPipeline``. Every feature uses only one epoch's own samples, so it is
leakage-free. Supervised, learned transforms (e.g. CSP) are out of scope.
"""

from eeg_decoder.features._array import FeatureArray
from eeg_decoder.features.spectral import CLASSIC_BANDS, band_power, welch_psd
from eeg_decoder.features.temporal import hjorth, time_stats
from eeg_decoder.features.tfr import morlet_tfr

__all__ = [
    "CLASSIC_BANDS",
    "FeatureArray",
    "band_power",
    "hjorth",
    "morlet_tfr",
    "time_stats",
    "welch_psd",
]
