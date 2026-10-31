"""Optional ICA-based artifact correction (blinks, eye movements, heartbeat).

Scientific background
---------------------
Independent component analysis unmixes the EEG channels into statistically
independent sources. Ocular and cardiac artifacts usually load on a few
components, which can be removed and the data reconstructed without them,
keeping the trials that threshold rejection would discard.

The workflow is split into three explicit steps, so the decision of which
components to remove is visible and reviewable:

1. :func:`fit_ica` estimates the unmixing (seeded, label-free);
2. :func:`find_artifact_components` scores components against EOG/ECG
   channels and proposes indices (it changes nothing);
3. :func:`apply_ica` removes exactly the components you list.

Behaviour and recommendations (verified on MNE 1.13)
----------------------------------------------------
* **Methods.** ``"infomax"`` (extended Infomax, handles both sub- and
  super-Gaussian sources) needs only MNE and is the default. ``"picard"``
  (same model, faster) needs ``python-picard``; ``"fastica"`` needs
  scikit-learn: ``pip install eeg-decoder[ica]``.
* **High-pass for fitting.** Slow drifts degrade ICA, so by default it is fit
  on a 1 Hz high-passed *copy*; the solution is then applied to the data you
  pass to :func:`apply_ica` (e.g. 0.1 Hz high-passed for ERPs).
* **Bad channels** are left out of the fit and are not changed by
  :func:`apply_ica`. Recommended order: mark bad channels -> fit and apply
  ICA -> interpolate bad channels -> re-reference.
* **Rank.** After an average reference the data lose one dimension; use
  ``n_components`` at most the number of good EEG channels minus one.
* **Exclusion.** MNE's ``ICA.apply(exclude=...)`` removes the *union* of the
  given components and ``ica.exclude``; :func:`apply_ica` removes exactly the
  components passed, regardless of ``ica.exclude``.
* **Leakage.** ICA uses no labels. It is fitted per recording on all of its
  (non-``BAD``) data; for strictly separated train/test segments of one
  recording, fit on the training segment only and apply to both.
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import mne
import numpy as np
from mne.preprocessing import ICA

_METHODS: dict[str, dict[str, Any]] = {
    "infomax": {"extended": True},
    "picard": {"extended": True, "ortho": False},
    "fastica": {},
}
_REQUIRES = {"picard": ("picard", "python-picard"), "fastica": ("sklearn", "scikit-learn")}


def _good_eeg(info: mne.Info) -> list[str]:
    return [info["ch_names"][i] for i in mne.pick_types(info, eeg=True, exclude="bads")]


def fit_ica(
    raw: mne.io.BaseRaw,
    *,
    n_components: int | float,
    seed: int,
    method: str = "infomax",
    highpass_hz: float | None = 1.0,
    max_iter: int | str = "auto",
) -> ICA:
    """Fit ICA on the good EEG channels of a continuous recording.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous recording with bad channels marked. Not modified. Spans
        annotated ``BAD_*`` are left out of the fit.
    n_components : int or float
        Number of components (int, 2 .. number of good EEG channels), or the
        fraction of variance to keep (float in (0, 1)).
    seed : int
        Seed for the ICA initialisation; required for reproducibility.
    method : {"infomax", "picard", "fastica"}, default="infomax"
        ``"infomax"`` runs extended Infomax and needs only MNE.
    highpass_hz : float or None, default=1.0
        Fit on a copy high-passed at this frequency (skipped if the data are
        already high-passed at or above it). ``None`` fits on the data as given.
    max_iter : int or "auto", default="auto"
        Maximum iterations, passed to MNE.

    Returns
    -------
    ica : mne.preprocessing.ICA
        Fitted solution. ``ica.info["highpass"]`` records the high-pass of
        the data it was fitted on; ``ica.exclude`` is empty.

    Raises
    ------
    TypeError
        On wrong input types.
    ValueError
        On empty data, fewer than 2 good EEG channels, invalid
        ``n_components``, ``seed``, ``method``, or ``highpass_hz``.
    ImportError
        If ``method`` needs a package that is not installed.
    """
    if isinstance(raw, mne.BaseEpochs) or not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(f"fit_ica expects continuous data (mne.io.BaseRaw), got {type(raw).__name__}.")
    if raw.n_times == 0:
        raise ValueError("Recording is empty (0 samples).")
    good = _good_eeg(raw.info)
    if len(good) < 2:
        raise ValueError(f"Need at least 2 good EEG channels for ICA, found {len(good)}.")

    if isinstance(n_components, bool) or not isinstance(n_components, (int, float, np.number)):
        raise TypeError(f"n_components must be an int or float, got {type(n_components).__name__}.")
    if isinstance(n_components, (int, np.integer)):
        if not 2 <= n_components <= len(good):
            raise ValueError(
                f"n_components={n_components} must be between 2 and the number of good "
                f"EEG channels ({len(good)}); after an average reference use at most {len(good) - 1}."
            )
        n_components = int(n_components)
    elif not 0 < float(n_components) < 1:
        raise ValueError(f"A float n_components is a variance fraction in (0, 1), got {n_components}.")

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}.")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {tuple(_METHODS)}, got {method!r}.")
    if method in _REQUIRES and importlib.util.find_spec(_REQUIRES[method][0]) is None:
        raise ImportError(
            f"method={method!r} needs {_REQUIRES[method][1]}: pip install eeg-decoder[ica] "
            "(or use method='infomax', which needs only MNE)."
        )

    fit_raw = raw
    if highpass_hz is not None:
        if isinstance(highpass_hz, bool) or not isinstance(highpass_hz, (int, float, np.number)):
            raise TypeError(f"highpass_hz must be a number or None, got {type(highpass_hz).__name__}.")
        if not math.isfinite(highpass_hz) or highpass_hz <= 0:
            raise ValueError(f"highpass_hz must be finite and > 0, got {highpass_hz}.")
        if raw.info["highpass"] < highpass_hz:
            # Imported here: preprocessing imports artifacts (pipeline ops), so a
            # module-level import would be circular.
            from eeg_decoder.preprocessing.filtering import bandpass

            fit_raw = bandpass(raw, float(highpass_hz), None, picks="eeg")

    ica = ICA(
        n_components=n_components,
        method=method,
        fit_params=dict(_METHODS[method]),
        max_iter=max_iter,
        rng=int(seed),
        verbose="error",
    )
    ica.fit(fit_raw, picks="eeg", reject_by_annotation=True, verbose="error")
    return ica


@dataclass(frozen=True)
class ComponentReport:
    """Components proposed for removal, with the evidence. Changes nothing.

    Attributes
    ----------
    eog, ecg : tuple of int
        Component indices selected for the EOG / ECG channel(s).
    exclude : tuple of int
        Sorted union of ``eog`` and ``ecg``: the proposal for :func:`apply_ica`.
    scores : mapping of str to tuple of float
        Per reference channel, the **correlation** of every component with it
        (also when ``measure="zscore"``: z-scoring happens internally).
    measure : str
        ``"zscore"`` or ``"correlation"``.
    threshold : float
        The threshold used with ``measure``.
    """

    eog: tuple[int, ...]
    ecg: tuple[int, ...]
    exclude: tuple[int, ...]
    scores: Mapping[str, tuple[float, ...]]
    measure: str
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable copy."""
        return {
            "eog": list(self.eog),
            "ecg": list(self.ecg),
            "exclude": list(self.exclude),
            "scores": {k: list(v) for k, v in self.scores.items()},
            "measure": self.measure,
            "threshold": self.threshold,
        }


def _check_fitted(ica: object) -> ICA:
    if not isinstance(ica, ICA):
        raise TypeError(f"Expected mne.preprocessing.ICA, got {type(ica).__name__}.")
    if ica.current_fit == "unfitted":
        raise ValueError("ICA is not fitted; use fit_ica first.")
    return ica


def find_artifact_components(
    ica: ICA,
    raw: mne.io.BaseRaw,
    *,
    eog: Sequence[str] | None = None,
    ecg: str | None = None,
    measure: str = "zscore",
    threshold: float = 3.0,
) -> ComponentReport:
    """Propose ocular and cardiac components by correlation with reference channels.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted solution (from :func:`fit_ica`). Not modified.
    raw : mne.io.BaseRaw
        Continuous recording containing the reference channels. Not modified.
    eog : list of str or None
        EOG channels (or frontal EEG channels as a proxy, e.g. ``["Fp1"]``).
        MNE band-passes them 1-10 Hz before correlating.
    ecg : str or None
        ECG channel name. MNE band-passes it 8-16 Hz before correlating.
    measure : {"zscore", "correlation"}, default="zscore"
        ``"zscore"`` (MNE's default): a component is selected when the
        z-score of its |correlation| across components exceeds
        ``threshold`` (iterated twice, MNE's rule). ``"correlation"``: when
        its |correlation| itself exceeds ``threshold`` (in (0, 1)), which
        does not depend on the number of components.
    threshold : float, default=3.0
        Threshold for ``measure``.

    Returns
    -------
    ComponentReport
        Proposed indices and scores. Nothing is removed until you call
        :func:`apply_ica` with the indices you accept.

    Raises
    ------
    TypeError
        On wrong input types.
    ValueError
        If neither ``eog`` nor ``ecg`` is given, a channel is missing, the
        ICA is unfitted, ``measure`` is unknown, or ``threshold`` is invalid
        for ``measure`` (including a z-score that cannot be reached, below).

    Notes
    -----
    With ``n`` components, the largest z-score any single component can
    reach is ``sqrt(n - 1)``, and MNE selects strictly above the threshold
    (verified on MNE 1.13). So the default 3.0 can never select anything
    with 10 or fewer components, however clear the artifact. That case
    raises instead of silently returning no components; use more components
    or ``measure="correlation"``.

    MNE's z-score rule is *iterated*: after removing the selected components
    it z-scores the rest again. With a low threshold this can add components
    whose correlation is near zero (verified: |r| = 0.09 selected at
    threshold 2.5 next to the true blink component). Always review
    ``scores`` before applying; ``measure="correlation"`` avoids this.
    """
    ica = _check_fitted(ica)
    if isinstance(raw, mne.BaseEpochs) or not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(f"Expected mne.io.BaseRaw, got {type(raw).__name__}.")
    if eog is None and ecg is None:
        raise ValueError("Give eog=[...] and/or ecg='...'.")
    if eog is not None:
        if isinstance(eog, str) or not isinstance(eog, Sequence) or not eog:
            raise TypeError("eog must be a non-empty list of channel names, e.g. ['EOG'].")
        eog = list(eog)
    if ecg is not None and not isinstance(ecg, str):
        raise TypeError("ecg must be a single channel name.")
    if measure not in ("zscore", "correlation"):
        raise ValueError(f"measure must be 'zscore' or 'correlation', got {measure!r}.")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.number)):
        raise TypeError("threshold must be a number.")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"threshold must be finite and > 0, got {threshold}.")
    if measure == "correlation" and threshold >= 1:
        raise ValueError(f"A correlation threshold must be in (0, 1), got {threshold}.")
    if measure == "zscore":
        z_max = math.sqrt(ica.n_components_ - 1)
        if threshold >= z_max:
            raise ValueError(
                f"With {ica.n_components_} components the largest reachable z-score is "
                f"sqrt({ica.n_components_} - 1) = {z_max:.2f}, so threshold={threshold:g} "
                "can never select a component. Fit more components, lower the "
                "threshold, or use measure='correlation' (e.g. threshold=0.7)."
            )
    wanted = (eog or []) + ([ecg] if ecg else [])
    missing = [c for c in wanted if c not in raw.ch_names]
    if missing:
        raise ValueError(f"Reference channel(s) not found: {missing}.")

    work = ica.copy()  # MNE's find_bads_* store labels on the ICA object
    scores: dict[str, tuple[float, ...]] = {}
    eog_idx: list[int] = []
    ecg_idx: list[int] = []
    if eog:
        eog_idx, sc = work.find_bads_eog(
            raw, ch_name=eog, threshold=threshold, measure=measure, verbose="error"
        )
        sc = np.atleast_2d(sc)
        scores.update({ch: tuple(map(float, row)) for ch, row in zip(eog, sc)})
    if ecg:
        ecg_idx, sc = work.find_bads_ecg(
            raw, ch_name=ecg, threshold=threshold, method="correlation",
            measure=measure, verbose="error",
        )
        scores[ecg] = tuple(map(float, np.ravel(sc)))
    eog_t = tuple(sorted(int(i) for i in eog_idx))
    ecg_t = tuple(sorted(int(i) for i in ecg_idx))
    return ComponentReport(
        eog=eog_t,
        ecg=ecg_t,
        exclude=tuple(sorted(set(eog_t) | set(ecg_t))),
        scores=MappingProxyType(scores),
        measure=measure,
        threshold=float(threshold),
    )


def apply_ica(
    inst: mne.io.BaseRaw | mne.BaseEpochs,
    ica: ICA,
    *,
    exclude: Sequence[int],
) -> mne.io.BaseRaw | mne.BaseEpochs:
    """Remove exactly the listed ICA components from a copy of the data.

    Parameters
    ----------
    inst : mne.io.BaseRaw or mne.BaseEpochs
        Data containing the channels the ICA was fitted on. Not modified.
    ica : mne.preprocessing.ICA
        Fitted solution. Not modified; its ``exclude`` attribute is ignored.
    exclude : sequence of int
        Component indices to remove, e.g. ``report.exclude``. Required, so
        the choice is always explicit; an empty list returns an unchanged copy.

    Returns
    -------
    inst_clean : same type as ``inst``
        Copy with the components removed from the fitted channels. Bad
        channels and channels outside the fit are unchanged.

    Raises
    ------
    TypeError
        On wrong input types.
    ValueError
        If the ICA is unfitted, or an index is duplicated or out of range.
    """
    ica = _check_fitted(ica)
    if not isinstance(inst, (mne.io.BaseRaw, mne.BaseEpochs)):
        raise TypeError(f"Expected mne.io.BaseRaw or mne.Epochs, got {type(inst).__name__}.")
    if isinstance(exclude, (str, bytes)) or not isinstance(exclude, Sequence):
        raise TypeError("exclude must be a list of component indices, e.g. [0, 3].")
    exclude = list(exclude)
    if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in exclude):
        raise TypeError("exclude must contain integer component indices.")
    if len(set(exclude)) != len(exclude):
        raise ValueError(f"Duplicate component indices: {exclude}.")
    bad = [i for i in exclude if not 0 <= i < ica.n_components_]
    if bad:
        raise ValueError(f"Component indices {bad} out of range 0..{ica.n_components_ - 1}.")

    out = inst.copy().load_data()
    if not exclude:
        return out
    work = ica.copy()
    work.exclude = []  # MNE would otherwise remove the union with ica.exclude
    work.apply(out, exclude=[int(i) for i in exclude], verbose="error")
    return out
