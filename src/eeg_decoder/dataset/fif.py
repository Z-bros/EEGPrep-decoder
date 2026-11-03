"""Saving and loading per-recording epochs as MNE ``.fif`` files with a JSON sidecar.

Unlike the ``.npz`` archive, a ``.fif`` file keeps the full MNE ``Epochs``
object (channel positions, filter settings, reference flag, baseline, events,
``drop_log``), so it can be reopened in MNE or any MNE-based tool. The JSON
sidecar next to it holds what MNE does not: the recording's metadata (as you
passed it) and its processing record.

Behaviour verified on MNE 1.13, and how this module handles it
----------------------------------------------------------------
* ``Epochs.save`` defaults to **single precision** (float32), so data reload
  only approximately. Files are written with ``fmt="double"``: the data
  round-trip exactly (at about twice the file size).
* A plain save keeps the **measurement date and subject information** (name,
  birthday, hospital ID) from the recording. By default the copy that is
  written is anonymized with ``Epochs.anonymize()`` (measurement date set to
  2000-01-01, subject fields replaced). Pass ``anonymize=False`` to keep them.
* MNE does not enforce its ``-epo.fif`` naming convention; it is required here.
* Existing files are never overwritten unless ``overwrite=True``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mne

from eeg_decoder.dataset.core import _check_metadata

_FORMAT = "eegprep-epochs-sidecar"
_VERSION = 1
_SUFFIX = "-epo.fif"


def _paths(path: str | Path) -> tuple[Path, Path]:
    if not isinstance(path, (str, Path)):
        raise TypeError(f"path must be a str or pathlib.Path, got {type(path).__name__}.")
    path = Path(path)
    if not path.name.endswith(_SUFFIX):
        raise ValueError(f"path must end in {_SUFFIX!r} (MNE's convention), got {path.name!r}.")
    return path, path.with_name(path.name[: -len(".fif")] + ".json")


def save_epochs_fif(
    epochs: mne.BaseEpochs,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    record: Mapping[str, Any] | None = None,
    anonymize: bool = True,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write epochs to ``<name>-epo.fif`` plus a ``<name>-epo.json`` sidecar.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epochs of one recording (at least one epoch). Not modified.
    path : str or pathlib.Path
        Destination ending in ``-epo.fif``; its directory must exist.
    metadata : mapping or None
        This recording's identifiers, e.g. ``{"subject": "01", "session": 1}``
        (str/int/float/bool values), stored in the sidecar.
    record : mapping or None
        JSON-serializable processing record, e.g. from :meth:`EEGPipeline.run`.
    anonymize : bool, default=True
        Remove the measurement date and subject information from the written
        copy (see module notes). The input is never modified.
    overwrite : bool, default=False
        Replace existing files. Otherwise an existing ``.fif`` or sidecar raises.

    Returns
    -------
    (fif_path, sidecar_path) : tuple of pathlib.Path

    Raises
    ------
    TypeError, ValueError
        On wrong inputs, an empty epochs object, a path not ending in
        ``-epo.fif``, invalid metadata, or a non-serializable record.
    FileExistsError
        If either file exists and ``overwrite`` is False (nothing is written).
    """
    if not isinstance(epochs, mne.BaseEpochs):
        raise TypeError(f"Expected mne.Epochs, got {type(epochs).__name__}.")
    if len(epochs) == 0:
        raise ValueError("Epochs object contains no epochs.")
    for flag, value in (("anonymize", anonymize), ("overwrite", overwrite)):
        if not isinstance(value, bool):
            raise TypeError(f"{flag} must be True or False.")
    fif_path, json_path = _paths(path)
    meta = _check_metadata([metadata or {}], 1)[0]
    sidecar = {
        "format": _FORMAT,
        "format_version": _VERSION,
        "eeg_decoder_version": __import__("eeg_decoder").__version__,
        "fif": fif_path.name,
        "anonymized": anonymize,
        "metadata": meta,
        "record": None if record is None else dict(record),
    }
    try:
        text = json.dumps(sidecar, indent=2, allow_nan=False)
    except (TypeError, ValueError) as err:
        raise ValueError(f"record must be JSON-serializable: {err}") from None
    if not overwrite:
        existing = [str(p) for p in (fif_path, json_path) if p.exists()]
        if existing:
            raise FileExistsError(f"{existing} exist; pass overwrite=True to replace them.")

    out = epochs.copy().load_data()
    if anonymize:
        out.anonymize(verbose="error")
    out.save(fif_path, fmt="double", overwrite=overwrite, verbose="error")
    json_path.write_text(text + "\n", encoding="utf-8")
    return fif_path, json_path


def load_epochs_fif(path: str | Path) -> tuple[mne.BaseEpochs, dict[str, Any]]:
    """Read epochs written by :func:`save_epochs_fif`, with their sidecar.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``-epo.fif`` file. Its ``-epo.json`` sidecar must be next to it.

    Returns
    -------
    epochs : mne.Epochs
        Preloaded epochs.
    sidecar : dict
        ``metadata``, ``record``, ``anonymized``, and format information.
        Pass ``metadata`` and ``record`` to :func:`combine_epochs` to rebuild a
        dataset from several files.

    Raises
    ------
    TypeError, ValueError
        On a wrong path, or a sidecar that is not from :func:`save_epochs_fif`.
    FileNotFoundError
        If the ``.fif`` file or its sidecar is missing.
    """
    fif_path, json_path = _paths(path)
    if not fif_path.exists():
        raise FileNotFoundError(fif_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Sidecar {json_path} not found next to {fif_path.name}.")
    sidecar = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(sidecar, dict) or sidecar.get("format") != _FORMAT:
        raise ValueError(f"{json_path} is not an EEGPrep-decoder epochs sidecar.")
    if sidecar.get("format_version") != _VERSION:
        raise ValueError(f"Unsupported sidecar version {sidecar.get('format_version')!r}.")
    epochs = mne.read_epochs(fif_path, preload=True, verbose="error")
    return epochs, sidecar
