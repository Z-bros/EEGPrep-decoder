"""Saving and loading an :class:`EpochDataset` as a NumPy ``.npz`` archive.

The archive holds only plain arrays (numbers and fixed-width strings) plus
JSON text, so :func:`load_npz` reads it with ``allow_pickle=False``: loading a
file can never execute code. Every field round-trips exactly.

Stored arrays: ``X``, ``y``, ``recording``, ``times``, ``ch_names``,
``ch_types``, ``sfreq``, one ``group__<key>`` array per metadata key, and JSON
text for ``event_id``, per-recording metadata, processing records, and
failures, plus a format name and version.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from eeg_decoder.dataset.core import EpochDataset, _readonly

_FORMAT = "eegprep-epochdataset"
_VERSION = 1
_GROUP = "group__"


def _check_path(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)):
        raise TypeError(f"path must be a str or pathlib.Path, got {type(path).__name__}.")
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"path must end in '.npz', got {path.name!r}.")
    return path


def save_npz(dataset: EpochDataset, path: str | Path, *, overwrite: bool = False) -> Path:
    """Write ``dataset`` to a compressed ``.npz`` file.

    Parameters
    ----------
    dataset : EpochDataset
        The dataset to save.
    path : str or pathlib.Path
        Destination ending in ``.npz``. Its directory must exist.
    overwrite : bool, default=False
        Replace an existing file. Otherwise an existing file raises.

    Returns
    -------
    pathlib.Path
        The path written.

    Raises
    ------
    TypeError
        If ``dataset`` is not an :class:`EpochDataset` or ``path`` has the wrong type.
    ValueError
        If ``path`` does not end in ``.npz``.
    FileExistsError
        If the file exists and ``overwrite`` is False.
    """
    import eeg_decoder

    if not isinstance(dataset, EpochDataset):
        raise TypeError(f"Expected EpochDataset, got {type(dataset).__name__}.")
    path = _check_path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it.")
    arrays = {
        "format": np.array(_FORMAT),
        "format_version": np.array(_VERSION),
        "eeg_decoder_version": np.array(eeg_decoder.__version__),
        "X": dataset.X,
        "y": dataset.y,
        "recording": dataset.recording,
        "times": dataset.times,
        "ch_names": np.array(dataset.ch_names, dtype=str),
        "ch_types": np.array(dataset.ch_types, dtype=str),
        "sfreq": np.array(dataset.sfreq),
        "group_keys": np.array(list(dataset.groups), dtype=str),
        "event_id_json": np.array(json.dumps(dict(dataset.event_id))),
        "metadata_json": np.array(json.dumps([dict(m) for m in dataset.metadata])),
        "records_json": np.array(json.dumps(list(dataset._records_json))),
        "failures_json": np.array(json.dumps([dict(f) for f in dataset.failures])),
    }
    for key, values in dataset.groups.items():
        arrays[_GROUP + key] = values
    with open(path, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    return path


def load_npz(path: str | Path) -> EpochDataset:
    """Read an :class:`EpochDataset` written by :func:`save_npz`.

    Parameters
    ----------
    path : str or pathlib.Path
        A ``.npz`` file from :func:`save_npz`.

    Returns
    -------
    EpochDataset

    Raises
    ------
    TypeError, ValueError
        On a wrong path, or a file that is not an EpochDataset archive of a
        supported version.
    FileNotFoundError
        If the file does not exist.
    """
    path = _check_path(path)
    with np.load(path, allow_pickle=False) as z:
        if "format" not in z.files or str(z["format"]) != _FORMAT:
            raise ValueError(f"{path} is not an EpochDataset archive.")
        if int(z["format_version"]) != _VERSION:
            raise ValueError(f"Unsupported format version {int(z['format_version'])}.")
        keys = [str(k) for k in z["group_keys"]]
        return EpochDataset(
            X=_readonly(z["X"]),
            y=_readonly(z["y"]),
            event_id=MappingProxyType(json.loads(str(z["event_id_json"]))),
            recording=_readonly(z["recording"]),
            groups=MappingProxyType({k: _readonly(z[_GROUP + k]) for k in keys}),
            ch_names=tuple(str(c) for c in z["ch_names"]),
            ch_types=tuple(str(c) for c in z["ch_types"]),
            sfreq=float(z["sfreq"]),
            times=_readonly(z["times"]),
            metadata=tuple(MappingProxyType(m) for m in json.loads(str(z["metadata_json"]))),
            _records_json=tuple(json.loads(str(z["records_json"]))),
            failures=tuple(MappingProxyType(f) for f in json.loads(str(z["failures_json"]))),
        )
