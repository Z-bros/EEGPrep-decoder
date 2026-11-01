"""Signal-quality metrics and plots. Descriptive only: no pass/fail verdicts."""

from eeg_decoder.quality.metrics import (
    EpochsQuality,
    RecordingQuality,
    epochs_quality,
    recording_quality,
)
from eeg_decoder.quality.plots import (
    plot_channel_quality,
    plot_psd_comparison,
    plot_rejection_summary,
)

__all__ = [
    "EpochsQuality",
    "RecordingQuality",
    "epochs_quality",
    "plot_channel_quality",
    "plot_psd_comparison",
    "plot_rejection_summary",
    "recording_quality",
]
