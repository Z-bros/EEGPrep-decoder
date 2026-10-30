"""Artifact handling: detecting and removing contaminated data."""

from eeg_decoder.artifacts.rejection import reject_epochs

__all__ = ["reject_epochs"]
