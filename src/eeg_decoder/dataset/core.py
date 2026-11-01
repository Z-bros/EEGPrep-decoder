"""Combining preprocessed recordings into one documented dataset.

:class:`EpochDataset` holds everything a downstream tool needs, with no
modelling in this library: ``X`` ``(n_epochs, n_channels, n_times)`` in volts,
``y`` (event codes) and ``event_id``, one identifier per epoch for every
metadata key (e.g. ``groups["subject"]``, for grouped train/test splits by
subject or session), which recording each epoch came from, channel names and
types, sampling rate, epoch times, and every recording's processing record.

Privacy: only the metadata you pass is stored. The measurement date and
subject information inside ``raw.info`` are never copied (the pipeline record
omits them too).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import mne
import numpy as np

from eeg_decoder.dataset.consistency import _as_array, check_compatible
from eeg_decoder.preprocessing.pipeline import EEGPipeline, PreprocessingConfig

_SCALARS = (str, int, float, bool)


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.array(a, copy=True)
    a.setflags(write=False)
    return a


def _check_metadata(metadata: Sequence[Mapping[str, Any]], n: int) -> list[dict[str, Any]]:
    if isinstance(metadata, Mapping) or not isinstance(metadata, Sequence) or len(metadata) != n:
        raise ValueError(f"metadata must be a list with one mapping per recording ({n}).")
    checked = []
    for i, meta in enumerate(metadata):
        if not isinstance(meta, Mapping):
            raise TypeError(f"metadata[{i}] must be a mapping, got {type(meta).__name__}.")
        for key, value in meta.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"metadata[{i}] keys must be non-empty strings, got {key!r}.")
            if not isinstance(value, _SCALARS) or isinstance(value, np.generic):
                raise TypeError(f"metadata[{i}][{key!r}] must be a str, int, float, or bool, "
                                f"got {type(value).__name__}.")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"metadata[{i}][{key!r}] must be finite.")
        checked.append(dict(meta))
    keys = [set(m) for m in checked]
    if any(k != keys[0] for k in keys):
        raise ValueError("Every recording needs the same metadata keys; got "
                         + ", ".join(f"{i}: {sorted(k)}" for i, k in enumerate(keys)) + ".")
    return checked


@dataclass(frozen=True, eq=False)
class EpochDataset:
    """Epochs from one or more recordings, with labels, identifiers, and provenance.

    Attributes
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_times)
        Read-only epoch data in volts.
    y : ndarray of int, shape (n_epochs,)
        Event code of each epoch; names via ``event_id`` or :attr:`y_names`.
    event_id : mapping of str to int
        Condition names to codes, consistent across recordings.
    recording : ndarray of int, shape (n_epochs,)
        Index of the recording each epoch came from.
    groups : mapping of str to ndarray, each shape (n_epochs,)
        One identifier array per metadata key, e.g. ``groups["subject"]``.
    ch_names, ch_types : tuple of str
        Channel names and types, in data order.
    sfreq : float
        Sampling rate in Hz.
    times : ndarray, shape (n_times,)
        Epoch time axis in seconds.
    metadata : tuple of mapping
        The metadata of each recording, as passed.
    failures : tuple of mapping
        Recordings skipped by :func:`prepare_recordings` with
        ``on_error="collect"``: index, metadata, error type, and message.
    """

    X: np.ndarray
    y: np.ndarray
    event_id: Mapping[str, int]
    recording: np.ndarray
    groups: Mapping[str, np.ndarray]
    ch_names: tuple[str, ...]
    ch_types: tuple[str, ...]
    sfreq: float
    times: np.ndarray
    metadata: tuple[Mapping[str, Any], ...]
    _records_json: tuple[str, ...] = field(repr=False)
    failures: tuple[Mapping[str, Any], ...] = ()

    @property
    def records(self) -> list[dict[str, Any] | None]:
        """Each recording's processing record (fresh, editable copies)."""
        return [json.loads(r) for r in self._records_json]

    @property
    def n_epochs(self) -> int:
        return int(self.X.shape[0])

    @property
    def y_names(self) -> np.ndarray:
        """Condition name of each epoch."""
        inverse = {code: name for name, code in self.event_id.items()}
        return np.array([inverse[int(c)] for c in self.y], dtype=str)

    def summary(self) -> str:
        """Human-readable overview."""
        counts = {name: int(np.sum(self.y == code)) for name, code in self.event_id.items()}
        lines = [
            f"{self.n_epochs} epochs from {len(self.metadata)} recording(s): "
            f"{len(self.ch_names)} channels x {len(self.times)} samples at {self.sfreq:g} Hz "
            f"({self.times[0]:g} to {self.times[-1]:g} s)",
            "per condition: " + ", ".join(f"{k}={v}" for k, v in counts.items()),
        ]
        for key, values in self.groups.items():
            lines.append(f"{key}: {len(np.unique(values))} distinct values")
        if self.failures:
            lines.append(f"{len(self.failures)} recording(s) failed and were skipped")
        return "\n".join(lines)

    def to_dataframe(self):
        """Per-epoch table (recording, condition, code, and every metadata key).

        Requires pandas, which is not a dependency of this library.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("to_dataframe needs pandas: pip install pandas") from None
        table = {"recording": self.recording, "condition": self.y_names, "code": self.y}
        table.update({k: v for k, v in self.groups.items()})
        return pd.DataFrame(table)

    def __repr__(self) -> str:
        return (f"EpochDataset(n_epochs={self.n_epochs}, n_channels={len(self.ch_names)}, "
                f"n_times={len(self.times)}, sfreq={self.sfreq:g}, "
                f"recordings={len(self.metadata)}, groups={list(self.groups)})")


def combine_epochs(
    epochs_list: Sequence[mne.BaseEpochs],
    *,
    metadata: Sequence[Mapping[str, Any]] | None = None,
    records: Sequence[Mapping[str, Any] | None] | None = None,
    ch_types: Sequence[str] | None = None,
) -> EpochDataset:
    """Combine compatible epochs from several recordings into an :class:`EpochDataset`.

    Parameters
    ----------
    epochs_list : sequence of mne.BaseEpochs
        One epochs object per recording; must pass :func:`check_compatible`
        (after ``ch_types`` selection). Not modified.
    metadata : sequence of mapping or None
        One mapping per recording (e.g. ``{"subject": "01", "session": "T"}``),
        with the same keys everywhere and str/int/float/bool values. Each key
        becomes a per-epoch identifier in ``groups``.
    records : sequence of (mapping or None) or None
        Optional JSON-serializable processing record per recording, e.g. from
        :meth:`EEGPipeline.run`.
    ch_types : sequence of str or None
        Keep only channels of these types in ``X``, e.g. ``["eeg"]`` to leave
        out stim and EOG channels. ``None`` (default) keeps every channel, so
        nothing is dropped unless you ask.

    Returns
    -------
    EpochDataset

    Raises
    ------
    TypeError, ValueError
        On incompatible recordings (all differences listed), malformed
        metadata or ``ch_types``, or non-serializable records.
    """
    if isinstance(epochs_list, mne.BaseEpochs) or not isinstance(epochs_list, Sequence):
        raise TypeError("epochs_list must be a list of mne.Epochs, one per recording.")
    if ch_types is not None:
        if isinstance(ch_types, str) or not isinstance(ch_types, Sequence) or not ch_types:
            raise TypeError("ch_types must be a non-empty list of channel types, e.g. ['eeg'].")
        wanted = list(ch_types)
        picked = []
        for i, ep in enumerate(epochs_list):
            if not isinstance(ep, mne.BaseEpochs):
                raise TypeError(f"Item {i} is {type(ep).__name__}, not mne.Epochs.")
            if not set(ep.get_channel_types()) & set(wanted):
                raise ValueError(f"recording {i} has no channels of type {wanted}.")
            picked.append(ep.copy().pick(wanted))
        epochs_list = picked
    labels = None
    if metadata is not None:
        metadata = _check_metadata(metadata, len(epochs_list))
        labels = [", ".join(f"{k}={v}" for k, v in m.items()) or str(i) for i, m in enumerate(metadata)]
    event_id = check_compatible(epochs_list, labels=labels)
    n = len(epochs_list)
    metadata = metadata if metadata is not None else [{} for _ in range(n)]
    if records is None:
        records = [None] * n
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or len(records) != n:
        raise ValueError(f"records must be a list with one entry per recording ({n}).")
    try:
        records_json = tuple(json.dumps(r, allow_nan=False) for r in records)
    except (TypeError, ValueError) as err:
        raise ValueError(f"records must be JSON-serializable: {err}") from None

    first = epochs_list[0]
    X = np.concatenate([ep.get_data() for ep in epochs_list], axis=0)
    y = np.concatenate([ep.events[:, 2] for ep in epochs_list]).astype(np.int64)
    sizes = [len(ep) for ep in epochs_list]
    recording = np.repeat(np.arange(n, dtype=np.int64), sizes)
    groups = {
        key: _readonly(_as_array([m[key] for m, k in zip(metadata, sizes) for _ in range(k)], key))
        for key in metadata[0]
    }
    return EpochDataset(
        X=_readonly(X),
        y=_readonly(y),
        event_id=MappingProxyType(dict(event_id)),
        recording=_readonly(recording),
        groups=MappingProxyType(groups),
        ch_names=tuple(first.ch_names),
        ch_types=tuple(first.get_channel_types()),
        sfreq=float(first.info["sfreq"]),
        times=_readonly(first.times),
        metadata=tuple(MappingProxyType(dict(m)) for m in metadata),
        _records_json=records_json,
    )


def prepare_recordings(
    recordings: Sequence[tuple[mne.io.BaseRaw, Mapping[str, Any]]],
    config: PreprocessingConfig,
    *,
    on_error: str = "raise",
    ch_types: Sequence[str] | None = None,
) -> EpochDataset:
    """Run one preprocessing config on several recordings and combine the epochs.

    Parameters
    ----------
    recordings : sequence of (mne.io.BaseRaw, mapping)
        ``(raw, metadata)`` pairs, e.g. ``(raw, {"subject": "01", "session": 1})``.
        Every metadata mapping needs the same keys. Recordings are not modified.
    config : PreprocessingConfig
        Must contain a ``make_epochs`` step, so each recording yields epochs.
    on_error : {"raise", "collect"}, default="raise"
        ``"raise"``: stop at the first failing recording; the error gets a
        note naming the recording. ``"collect"``: skip failing recordings
        (including ones with no epochs left) and list them in
        ``dataset.failures``. Incompatibilities *between* successful
        recordings always raise.
    ch_types : sequence of str or None
        Passed to :func:`combine_epochs`, e.g. ``["eeg"]``; ``None`` keeps all.

    Returns
    -------
    EpochDataset
        Combined epochs with per-epoch identifiers, and each recording's
        pipeline record in ``dataset.records``.

    Raises
    ------
    TypeError, ValueError
        On malformed inputs, a config without ``make_epochs``, incompatible
        results, or (with ``"collect"``) when every recording fails. With
        ``"raise"``, the first recording's error propagates.
    """
    if not isinstance(config, PreprocessingConfig):
        raise TypeError(f"config must be a PreprocessingConfig, got {type(config).__name__}.")
    if "make_epochs" not in config.ops:
        raise ValueError("config must contain a 'make_epochs' step to build a dataset of epochs.")
    if on_error not in ("raise", "collect"):
        raise ValueError(f"on_error must be 'raise' or 'collect', got {on_error!r}.")
    if isinstance(recordings, Mapping) or not isinstance(recordings, Sequence) or not recordings:
        raise TypeError("recordings must be a non-empty list of (raw, metadata) pairs.")
    for i, item in enumerate(recordings):
        if not (isinstance(item, tuple) and len(item) == 2):
            raise TypeError(f"recordings[{i}] must be a (raw, metadata) pair.")
    metadata = _check_metadata([meta for _, meta in recordings], len(recordings))

    pipeline = EEGPipeline(config)
    kept_index, kept_epochs, kept_meta, kept_records, failures = [], [], [], [], []
    for i, ((raw, _), meta) in enumerate(zip(recordings, metadata)):
        try:
            epochs, record = pipeline.run(raw, metadata=meta)
            if len(epochs) == 0:
                raise ValueError("no epochs remain after preprocessing")
        except Exception as exc:
            if on_error == "raise":
                exc.add_note(f"While preparing recording {i} with metadata {meta}.")
                raise
            failures.append(MappingProxyType({
                "index": i, "metadata": dict(meta),
                "error": type(exc).__name__, "message": str(exc),
            }))
            continue
        kept_index.append(i)
        kept_epochs.append(epochs)
        kept_meta.append(meta)
        kept_records.append(record)

    if not kept_epochs:
        raise ValueError("Every recording failed:\n  " + "\n  ".join(
            f"{f['index']} {dict(f['metadata'])}: {f['error']}: {f['message']}" for f in failures))
    dataset = combine_epochs(kept_epochs, metadata=kept_meta, records=kept_records,
                             ch_types=ch_types)
    # Index everything by position in the *input* list, so a skipped recording
    # does not shift the others: `recording`, `metadata`, and `records` all use
    # input positions, and a failed recording's record is None.
    record_by_index = dict(zip(kept_index, dataset._records_json))
    return replace(
        dataset,
        recording=_readonly(np.asarray(kept_index, dtype=np.int64)[dataset.recording]),
        metadata=tuple(MappingProxyType(dict(m)) for m in metadata),
        _records_json=tuple(record_by_index.get(i, "null") for i in range(len(recordings))),
        failures=tuple(failures),
    )
