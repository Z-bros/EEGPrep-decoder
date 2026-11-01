"""Dataset preparation: combining recordings, preserving metadata, exporting.

No labels are used to change any data, and nothing is fitted: identifiers
are carried along only so downstream code can make grouped splits (by
subject, session, ...). Only metadata you pass is stored.
"""

from eeg_decoder.dataset.consistency import check_compatible, match_channels
from eeg_decoder.dataset.core import EpochDataset, combine_epochs, prepare_recordings
from eeg_decoder.dataset.export import load_npz, save_npz

__all__ = [
    "EpochDataset",
    "check_compatible",
    "combine_epochs",
    "load_npz",
    "match_channels",
    "prepare_recordings",
    "save_npz",
]
