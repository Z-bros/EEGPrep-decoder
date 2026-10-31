"""Artifact handling: detecting and removing contaminated data."""

from eeg_decoder.artifacts.annotations import add_annotations, annotate_artifacts
from eeg_decoder.artifacts.channels import (
    ChannelAssessment,
    assess_channels,
    interpolate_bad_channels,
    mark_bad_channels,
)
from eeg_decoder.artifacts.ica import (
    ComponentReport,
    apply_ica,
    find_artifact_components,
    fit_ica,
)
from eeg_decoder.artifacts.rejection import reject_epochs

__all__ = [
    "ChannelAssessment",
    "ComponentReport",
    "add_annotations",
    "annotate_artifacts",
    "apply_ica",
    "assess_channels",
    "find_artifact_components",
    "fit_ica",
    "interpolate_bad_channels",
    "mark_bad_channels",
    "reject_epochs",
]
