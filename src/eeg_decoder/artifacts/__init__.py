"""Artifact handling: detecting and removing contaminated data."""

from eeg_decoder.artifacts.annotations import add_annotations, annotate_artifacts
from eeg_decoder.artifacts.channels import (
    ChannelAssessment,
    assess_channels,
    interpolate_bad_channels,
    mark_bad_channels,
)
from eeg_decoder.artifacts.rejection import reject_epochs

__all__ = [
    "ChannelAssessment",
    "add_annotations",
    "annotate_artifacts",
    "assess_channels",
    "interpolate_bad_channels",
    "mark_bad_channels",
    "reject_epochs",
]
