"""EEG re-referencing.

Scientific background
---------------------
EEG measures potential *differences*: every channel is recorded relative to a
reference electrode chosen by the hardware. Re-referencing expresses all EEG
channels relative to a different reference:

* **Average reference** subtracts, at every sample, the mean of all good EEG
  channels. It approximates a neutral reference only with dense, roughly
  uniform scalp coverage; with few channels it mostly mixes their signals.
* **Channel reference** subtracts one channel (e.g. ``["Cz"]``) or the mean of
  several (e.g. linked mastoids ``["M1", "M2"]``). The reference channels
  become flat (all zeros for a single channel) and are kept, not dropped.

Re-referencing is a linear, sample-by-sample spatial operation, so it is valid
on continuous data and on epochs, and gives the same result in either order.

Behaviour verified on MNE 1.13
------------------------------
* Bad channels are left out of the reference **and are not re-referenced
  themselves**: they stay in the old reference. The recording then mixes two
  references, so this function refuses data with bad EEG channels unless
  ``allow_bads=True``. Interpolate (``inst.interpolate_bads()``, needs
  electrode positions) or drop them first.
* Only EEG channels change. EOG, ECG, EMG, stim, and misc are untouched.
* The reference is applied to the data directly (no deferred projector).
  ``info["custom_ref_applied"]`` is set, which records *that* a custom
  reference was applied but not *which* one.
"""

from __future__ import annotations

from collections.abc import Sequence

import mne


def _eeg_channels(info: mne.Info) -> list[str]:
    return [info["ch_names"][i] for i in mne.pick_types(info, eeg=True, exclude=[])]


def rereference(
    inst: mne.io.BaseRaw | mne.BaseEpochs,
    ref: str | Sequence[str],
    *,
    allow_bads: bool = False,
) -> mne.io.BaseRaw | mne.BaseEpochs:
    """Re-reference EEG channels to the average or to specific channels.

    Parameters
    ----------
    inst : mne.io.BaseRaw or mne.BaseEpochs
        Continuous or epoched data. Not modified.
    ref : "average" or sequence of str
        ``"average"`` for the common average of good EEG channels, or a list
        of EEG channel names whose mean becomes the new reference, e.g.
        ``["Cz"]`` or ``["M1", "M2"]``. There is no default: the choice
        changes the meaning of every channel and must be explicit.
    allow_bads : bool, default=False
        If False, raise when any EEG channel is listed in ``info["bads"]``
        (see module notes). If True, follow MNE: bad channels are excluded
        from the reference and left in the old reference.

    Returns
    -------
    inst_ref : same type as ``inst``
        Re-referenced copy. Shape, sampling rate, channel names, channel
        types, and ``info["bads"]`` are unchanged.

    Raises
    ------
    TypeError
        If ``inst`` is not MNE Raw/Epochs, or ``ref`` has the wrong type.
    ValueError
        If the data are empty or contain no EEG channels; if ``ref`` names a
        channel that is missing, not EEG, duplicated, or marked bad; if the
        average reference has fewer than 2 good EEG channels; or if bad EEG
        channels are present and ``allow_bads`` is False.

    Notes
    -----
    A reference electrode that was not recorded (for example the online
    reference of many amplifiers) is not in the data, so it cannot take part
    in an average reference. Restoring it (``mne.add_reference_channels``) is
    not handled here.
    """
    if not isinstance(inst, (mne.io.BaseRaw, mne.BaseEpochs)):
        raise TypeError(
            f"Expected mne.io.BaseRaw or mne.Epochs, got {type(inst).__name__}. "
            "Convert arrays with eeg_decoder.io.to_raw first."
        )
    if isinstance(inst, mne.io.BaseRaw) and inst.n_times == 0:
        raise ValueError("Recording is empty (0 samples); nothing to re-reference.")

    eeg = _eeg_channels(inst.info)
    if not eeg:
        raise ValueError("No EEG channels found; re-referencing applies to EEG only.")
    bads = [c for c in inst.info["bads"] if c in eeg]
    good = [c for c in eeg if c not in bads]

    if isinstance(ref, str):
        if ref != "average":
            raise ValueError(
                f"ref must be 'average' or a list of channel names, got {ref!r}. "
                f"For a single reference channel use ref=[{ref!r}]."
            )
        if len(good) < 2:
            raise ValueError(
                f"Average reference needs at least 2 good EEG channels, found {len(good)}."
            )
        ref_channels: str | list[str] = "average"
    else:
        try:
            ref_channels = list(ref)
        except TypeError:
            raise TypeError(
                f"ref must be 'average' or a sequence of str, got {type(ref).__name__}."
            ) from None
        if not ref_channels:
            raise ValueError("ref must name at least one channel.")
        if not all(isinstance(c, str) for c in ref_channels):
            raise TypeError("ref channel names must be strings.")
        if len(set(ref_channels)) != len(ref_channels):
            raise ValueError(f"Duplicate reference channels: {ref_channels}.")
        missing = [c for c in ref_channels if c not in inst.ch_names]
        if missing:
            raise ValueError(f"Reference channel(s) not found: {missing}.")
        not_eeg = [c for c in ref_channels if c not in eeg]
        if not_eeg:
            raise ValueError(f"Reference channel(s) are not EEG: {not_eeg}.")
        bad_refs = [c for c in ref_channels if c in bads]
        if bad_refs:
            raise ValueError(f"Reference channel(s) are marked bad: {bad_refs}.")

    if bads and not allow_bads:
        raise ValueError(
            f"Bad EEG channels {bads} would be excluded from the reference and left "
            "in the old reference (MNE behaviour), mixing two references. "
            "Interpolate them (inst.interpolate_bads()) or drop them first, or pass "
            "allow_bads=True to accept this."
        )

    out = inst.copy().load_data()
    if isinstance(out, mne.BaseEpochs) and len(out) == 0:
        raise ValueError("Epochs object contains no epochs; nothing to re-reference.")
    out.set_eeg_reference(ref_channels, projection=False, ch_type="eeg", verbose="error")
    return out
