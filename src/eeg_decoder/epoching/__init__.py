"""Cutting continuous recordings into event-locked epochs."""

from eeg_decoder.epoching.epochs import baseline_correct, make_epochs

__all__ = ["baseline_correct", "make_epochs"]
