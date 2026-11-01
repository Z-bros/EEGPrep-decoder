"""Checking that several recordings can be combined into one dataset.

Combining epochs from different recordings is only meaningful when every
recording has the same channels in the same order, the same sampling rate and
epoch window, the same meaning of each event code, the same filters and
reference, and the same units. MNE's ``mne.concatenate_epochs`` already
refuses some mismatches, but stops at the first one (verified on MNE 1.13).
:func:`check_compatible` lists *all* of them at once, and nothing here ever
resamples, re-references, interpolates, or reorders data silently.
"""

from __future__ import annotations

from collections.abc import Sequence

import mne
import numpy as np


def _describe(index: int, labels: Sequence[str] | None) -> str:
    return f"recording {index}" + (f" ({labels[index]})" if labels else "")


def check_compatible(
    epochs_list: Sequence[mne.BaseEpochs],
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, int]:
    """Verify that epochs from several recordings can be combined.

    Every recording is compared with the first one on: channel names and
    order, channel types, channel units, sampling rate, epoch start and
    length, filter edges (``highpass``/``lowpass``), whether the reference was
    changed, baseline window, and bad channels (none are allowed: interpolate
    or drop them first). Event codes must mean the same thing everywhere: a
    condition name maps to one code and a code to one name, though not every
    recording needs every condition. Empty recordings are rejected.

    Parameters
    ----------
    epochs_list : sequence of mne.BaseEpochs
        One epochs object per recording. Not modified.
    labels : sequence of str or None
        Optional names for the recordings, used in messages.

    Returns
    -------
    event_id : dict of str to int
        The union of all ``event_id`` mappings.

    Raises
    ------
    TypeError
        If ``epochs_list`` is not a non-empty sequence of MNE Epochs.
    ValueError
        Listing every incompatibility found, one per line.
    """
    if isinstance(epochs_list, mne.BaseEpochs) or not isinstance(epochs_list, Sequence):
        raise TypeError("epochs_list must be a list of mne.Epochs, one per recording.")
    if len(epochs_list) == 0:
        raise ValueError("epochs_list is empty.")
    for i, ep in enumerate(epochs_list):
        if not isinstance(ep, mne.BaseEpochs):
            raise TypeError(f"Item {i} is {type(ep).__name__}, not mne.Epochs.")
    if labels is not None and len(labels) != len(epochs_list):
        raise ValueError("labels must have one entry per recording.")

    ref = epochs_list[0]
    problems: list[str] = []

    def check(i: int, what: str, value: object, expected: object) -> None:
        if value != expected:
            problems.append(f"{_describe(i, labels)}: {what} is {value!r}, expected {expected!r} "
                            f"(as in {_describe(0, labels)}).")

    for i, ep in enumerate(epochs_list):
        name = _describe(i, labels)
        if len(ep) == 0:
            problems.append(f"{name}: contains no epochs.")
        if ep.info["bads"]:
            problems.append(f"{name}: has bad channels {list(ep.info['bads'])}; "
                            "interpolate or drop them before combining.")
        if i == 0:
            continue
        if ep.ch_names != ref.ch_names:
            if sorted(ep.ch_names) == sorted(ref.ch_names):
                problems.append(f"{name}: same channels in a different order; "
                                "use match_channels(epochs, ch_names) to reorder explicitly.")
            else:
                missing = [c for c in ref.ch_names if c not in ep.ch_names]
                extra = [c for c in ep.ch_names if c not in ref.ch_names]
                problems.append(f"{name}: channel set differs (missing {missing}, extra {extra}).")
        else:
            check(i, "channel types", ep.get_channel_types(), ref.get_channel_types())
            units = [(int(c["unit"]), int(c["unit_mul"])) for c in ep.info["chs"]]
            ref_units = [(int(c["unit"]), int(c["unit_mul"])) for c in ref.info["chs"]]
            check(i, "channel units", units, ref_units)
        check(i, "sampling rate (Hz)", float(ep.info["sfreq"]), float(ref.info["sfreq"]))
        check(i, "epoch start tmin (s)", round(float(ep.tmin), 9), round(float(ref.tmin), 9))
        check(i, "samples per epoch", len(ep.times), len(ref.times))
        check(i, "highpass (Hz)", float(ep.info["highpass"]), float(ref.info["highpass"]))
        check(i, "lowpass (Hz)", float(ep.info["lowpass"]), float(ref.info["lowpass"]))
        check(i, "custom reference applied", bool(ep.info["custom_ref_applied"]),
              bool(ref.info["custom_ref_applied"]))
        check(i, "baseline", ep.baseline, ref.baseline)

    union: dict[str, int] = {}
    code_owner: dict[int, str] = {}
    for i, ep in enumerate(epochs_list):
        for cond, code in ep.event_id.items():
            code = int(code)
            if cond in union and union[cond] != code:
                problems.append(f"{_describe(i, labels)}: condition {cond!r} has code {code}, "
                                f"but {union[cond]} elsewhere.")
            elif code in code_owner and code_owner[code] != cond:
                problems.append(f"{_describe(i, labels)}: code {code} means {cond!r}, "
                                f"but {code_owner[code]!r} elsewhere.")
            else:
                union.setdefault(cond, code)
                code_owner.setdefault(code, cond)

    if problems:
        raise ValueError(f"{len(problems)} incompatibilities between recordings:\n  "
                         + "\n  ".join(problems))
    return union


def match_channels(
    inst: mne.io.BaseRaw | mne.BaseEpochs,
    ch_names: Sequence[str],
) -> mne.io.BaseRaw | mne.BaseEpochs:
    """Return a copy containing exactly ``ch_names``, in that order.

    Parameters
    ----------
    inst : mne.io.BaseRaw or mne.BaseEpochs
        Data. Not modified.
    ch_names : sequence of str
        The channels to keep, in the desired order. Channels not listed are
        dropped (explicitly, by your list); none are added or interpolated.

    Returns
    -------
    inst_matched : same type as ``inst``

    Raises
    ------
    TypeError
        On wrong input types.
    ValueError
        If ``ch_names`` is empty, has duplicates, or names channels that are
        missing from ``inst``.
    """
    if not isinstance(inst, (mne.io.BaseRaw, mne.BaseEpochs)):
        raise TypeError(f"Expected mne.io.BaseRaw or mne.Epochs, got {type(inst).__name__}.")
    if isinstance(ch_names, str) or not isinstance(ch_names, Sequence):
        raise TypeError("ch_names must be a list of channel names.")
    ch_names = list(ch_names)
    if not ch_names:
        raise ValueError("ch_names must not be empty.")
    if not all(isinstance(c, str) for c in ch_names):
        raise TypeError("channel names must be strings.")
    if len(set(ch_names)) != len(ch_names):
        raise ValueError(f"Duplicate channel names: {ch_names}.")
    missing = [c for c in ch_names if c not in inst.ch_names]
    if missing:
        raise ValueError(f"Channel(s) not in the data: {missing}. Nothing is interpolated "
                         "or added here; interpolate first if needed.")
    out = inst.copy().pick(ch_names)
    out.reorder_channels(ch_names)
    out.info["bads"] = [c for c in out.ch_names if c in inst.info["bads"]]
    return out


def _as_array(values: list, key: str) -> np.ndarray:
    """Per-epoch identifier column with one consistent JSON-scalar type."""
    kinds = {type(v) for v in values}
    if kinds <= {str}:
        return np.array(values, dtype=str)
    if kinds <= {bool}:
        return np.array(values, dtype=bool)
    if kinds <= {int}:
        return np.array(values, dtype=np.int64)
    if kinds <= {int, float}:
        return np.array(values, dtype=np.float64)
    raise ValueError(f"metadata key {key!r} mixes value types {sorted(k.__name__ for k in kinds)}; "
                     "use one type (e.g. strings) for every recording.")
