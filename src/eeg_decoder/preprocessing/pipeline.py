"""A composable, serializable preprocessing pipeline.

A :class:`PreprocessingConfig` is an ordered list of steps. Each step is a
plain mapping ``{"op": <function name>, **keyword arguments}``, where ``op``
names one of the package's preprocessing functions and the remaining keys are
that function's own keyword arguments::

    config = PreprocessingConfig([
        {"op": "notch", "freqs": [50.0, 100.0]},
        {"op": "bandpass", "l_freq": 1.0, "h_freq": 40.0},
        {"op": "rereference", "ref": "average"},
        {"op": "resample", "sfreq": 125.0},
        {"op": "make_epochs", "event_source": "stim", "stim_channel": "STI",
         "event_id": {"left": 1, "right": 2}, "tmin": -0.2, "tmax": 0.8},
        {"op": "baseline_correct", "baseline": [None, 0.0]},
        {"op": "reject_epochs", "reject": {"eeg": 150e-6}},
    ])
    epochs, record = EEGPipeline(config).run(raw, metadata={"subject": "01"})

The JSON form of the config is exactly this list, so a config written in
Python and one loaded from disk are interchangeable.

Design notes
------------
* **Order is validated** when the config is built: continuous-only steps
  (``notch``, ``bandpass``, ``resample``) cannot follow ``make_epochs``;
  epoch-only steps (``baseline_correct``, ``reject_epochs``) cannot precede
  it; ``rereference`` works on both; ``make_epochs`` appears at most once.
* **Parameter names are validated** against each function's signature when
  the config is built. Parameter *values* (e.g. a cutoff above Nyquist) depend
  on the data, so the functions themselves check them during :meth:`run`.
* **Events are extracted inside the pipeline**, at the ``make_epochs`` step,
  from the data as it is at that point (i.e. after any resampling).
* **No step is fitted to data.** Every step is a fixed, per-recording or
  per-epoch operation, so running the pipeline before a train/test split
  leaks nothing. Learned transforms belong in the decoding stage.
* **The record** returned by :meth:`EEGPipeline.run` documents the config,
  package versions, user-supplied metadata, the input, and every step's
  output (sampling rate, size, dropped epochs). It deliberately omits the
  measurement date and subject info stored in ``raw.info``, which may be
  identifying; pass any identifiers you need through ``metadata``.
"""

from __future__ import annotations

import inspect
import json
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any

import mne
import numpy as np

from eeg_decoder.artifacts import reject_epochs
from eeg_decoder.epoching import baseline_correct, make_epochs
from eeg_decoder.preprocessing.filtering import bandpass, notch
from eeg_decoder.preprocessing.referencing import rereference
from eeg_decoder.preprocessing.resampling import resample

SCHEMA_VERSION = 1

_CONTINUOUS, _EPOCHED, _BOTH = "continuous", "epoched", "both"
_EVENT_SOURCES = ("stim", "annotations")


@dataclass(frozen=True)
class _Op:
    func: Callable[..., Any]
    stage: str  # where the step may run: continuous, epoched, or both
    pipeline_params: tuple[str, ...] = ()  # handled by the pipeline, not the function
    supplied_params: tuple[str, ...] = ()  # passed by the pipeline, not allowed in config


_OPS: dict[str, _Op] = {
    "notch": _Op(notch, _CONTINUOUS),
    "bandpass": _Op(bandpass, _CONTINUOUS),
    "rereference": _Op(rereference, _BOTH),
    "resample": _Op(resample, _CONTINUOUS),
    "make_epochs": _Op(
        make_epochs, _CONTINUOUS, ("event_source", "stim_channel"), ("events",)
    ),
    "baseline_correct": _Op(baseline_correct, _EPOCHED),
    "reject_epochs": _Op(reject_epochs, _EPOCHED),
}


# --- immutability helpers ---------------------------------------------------


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    """Frozen -> plain JSON-like Python (dicts and lists)."""
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


def _json_check(value: Any, what: str) -> None:
    try:
        json.dumps(_thaw(value), allow_nan=False)
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"{what} must be JSON-serializable (str, int, float, bool, None, "
            f"lists, dicts with str keys; no NaN/Inf): {err}"
        ) from None


# --- validation -------------------------------------------------------------


def _check_step(index: int, step: object) -> tuple[str, dict[str, Any]]:
    where = f"Step {index}"
    if not isinstance(step, Mapping):
        raise TypeError(f"{where} must be a mapping like {{'op': 'bandpass', ...}}, got {step!r}.")
    if "op" not in step:
        raise ValueError(f"{where} has no 'op' key.")
    op = step["op"]
    if op not in _OPS:
        raise ValueError(f"{where}: unknown op {op!r}; available: {sorted(_OPS)}.")
    params = {k: v for k, v in step.items() if k != "op"}
    where = f"Step {index} ({op})"
    _json_check(params, f"{where} parameters")

    spec = _OPS[op]
    for name in spec.supplied_params:
        if name in params:
            raise ValueError(f"{where}: {name!r} is supplied by the pipeline, not the config.")
    call_params = {k: v for k, v in params.items() if k not in spec.pipeline_params}
    positional = [None] * (1 + len(spec.supplied_params))  # data (+ events)
    try:
        inspect.signature(spec.func).bind(*positional, **call_params)
    except TypeError as err:
        raise ValueError(f"{where}: invalid parameters ({err}).") from None

    if op == "make_epochs":
        source = params.get("event_source")
        if source not in _EVENT_SOURCES:
            raise ValueError(f"{where}: event_source must be one of {_EVENT_SOURCES}, got {source!r}.")
        if source == "stim" and not isinstance(params.get("stim_channel"), str):
            raise ValueError(f"{where}: event_source='stim' needs stim_channel (a channel name).")
        if source == "annotations" and "stim_channel" in params:
            raise ValueError(f"{where}: stim_channel is only used with event_source='stim'.")
    return op, params


def _order_error(ops: Sequence[str], start: str) -> str | None:
    """Return why ``ops`` cannot run from ``start``, or None if they can."""
    stage = start
    for i, op in enumerate(ops):
        if op == "make_epochs":
            if stage != _CONTINUOUS:
                return (f"Step {i} (make_epochs) needs continuous data, but the data "
                        "are already epoched at that point.")
            stage = _EPOCHED
            continue
        allowed = _OPS[op].stage
        if allowed not in (_BOTH, stage):
            return (f"Step {i} ({op}) runs on {allowed} data, but the data are {stage} "
                    "at that point. Order: continuous steps -> make_epochs -> epoch steps.")
    return None


# --- config -----------------------------------------------------------------


@dataclass(frozen=True, init=False)
class PreprocessingConfig:
    """Immutable, validated, JSON-serializable list of preprocessing steps.

    Parameters
    ----------
    steps : sequence of mapping
        Each ``{"op": name, **kwargs}``. ``op`` is one of ``notch``,
        ``bandpass``, ``rereference``, ``resample``, ``make_epochs``,
        ``baseline_correct``, ``reject_epochs``; the other keys are that
        function's keyword arguments. ``make_epochs`` additionally takes
        ``event_source`` (``"stim"`` or ``"annotations"``) and, for
        ``"stim"``, ``stim_channel``; its ``events`` are found by the pipeline.

    Raises
    ------
    TypeError
        If ``steps`` or a step has the wrong type.
    ValueError
        If ``steps`` is empty, an op is unknown, parameters do not match the
        function signature or are not JSON-serializable, or the step order is
        impossible for both continuous and epoched input.

    Notes
    -----
    Steps are stored frozen (mappings become read-only, lists become tuples),
    so a config cannot change after an experiment starts. :meth:`to_dict`
    returns plain, editable copies.
    """

    steps: tuple[Mapping[str, Any], ...]

    def __init__(self, steps: Sequence[Mapping[str, Any]]) -> None:
        if isinstance(steps, (str, bytes, Mapping)) or not isinstance(steps, Sequence):
            raise TypeError("steps must be a list of step mappings.")
        if len(steps) == 0:
            raise ValueError("steps must contain at least one step.")
        checked = [_check_step(i, s) for i, s in enumerate(steps)]
        ops = [op for op, _ in checked]
        if ops.count("make_epochs") > 1:
            raise ValueError("make_epochs may appear at most once.")
        err = _order_error(ops, _CONTINUOUS)
        if err is not None and _order_error(ops, _EPOCHED) is not None:
            raise ValueError(err)
        frozen = tuple(_freeze({"op": op, **params}) for op, params in checked)
        object.__setattr__(self, "steps", frozen)

    @property
    def ops(self) -> tuple[str, ...]:
        """Step names, in order."""
        return tuple(step["op"] for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict ``{"schema_version": 1, "steps": [...]}`` (JSON-ready)."""
        return {"schema_version": SCHEMA_VERSION, "steps": [_thaw(s) for s in self.steps]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PreprocessingConfig:
        """Build a config from :meth:`to_dict` output; unknown keys raise."""
        if not isinstance(data, Mapping):
            raise TypeError(f"Expected a mapping, got {type(data).__name__}.")
        if set(data) != {"schema_version", "steps"}:
            raise ValueError(
                f"Config keys must be exactly 'schema_version' and 'steps', got {sorted(data)}."
            )
        if data["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {data['schema_version']!r}; expected {SCHEMA_VERSION}."
            )
        return cls(data["steps"])

    def save(self, path: str | Path) -> None:
        """Write the config as JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> PreprocessingConfig:
        """Read a config written by :meth:`save`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --- pipeline ---------------------------------------------------------------


def _versions() -> dict[str, str]:
    import eeg_decoder

    out = {"python": platform.python_version(), "eeg_decoder": eeg_decoder.__version__}
    for pkg in ("mne", "numpy", "scipy"):
        out[pkg] = importlib_metadata.version(pkg)
    return out


def _summary(inst: mne.io.BaseRaw | mne.BaseEpochs) -> dict[str, Any]:
    info = inst.info
    out: dict[str, Any] = {
        "type": "epochs" if isinstance(inst, mne.BaseEpochs) else "raw",
        "sfreq": float(info["sfreq"]),
        "n_channels": len(inst.ch_names),
        "ch_names": list(inst.ch_names),
        "ch_types": list(inst.get_channel_types()),
        "bads": list(info["bads"]),
        "highpass": float(info["highpass"]),
        "lowpass": float(info["lowpass"]),
        "custom_ref_applied": bool(info["custom_ref_applied"]),
    }
    if isinstance(inst, mne.BaseEpochs):
        out.update(
            n_epochs=len(inst),
            n_times=len(inst.times),
            tmin=float(inst.tmin),
            tmax=float(inst.tmax),
            event_id={k: int(v) for k, v in inst.event_id.items()},
            baseline=None if inst.baseline is None else [float(b) for b in inst.baseline],
        )
    else:
        out.update(n_times=int(inst.n_times), duration_s=inst.n_times / info["sfreq"])
    return out


def _extract_events(
    raw: mne.io.BaseRaw, source: str, stim_channel: str | None, event_id: Mapping[str, int]
) -> np.ndarray:
    if source == "stim":
        if stim_channel not in raw.ch_names:
            raise ValueError(f"stim_channel {stim_channel!r} not found in the data.")
        if raw.get_channel_types(picks=[stim_channel])[0] != "stim":
            raise ValueError(f"stim_channel {stim_channel!r} is not a stim channel.")
        # shortest_event=1: one-sample pulses count; consecutive=True: a step
        # from one code directly to another is a new event.
        events = mne.find_events(
            raw, stim_channel=stim_channel, shortest_event=1, consecutive=True, verbose="error"
        )
        if events.shape[0] == 0:
            raise ValueError(f"No events found on stim channel {stim_channel!r}.")
        return events
    # MNE 1.13 raises a generic ValueError when annotations exist but none
    # match, and returns an empty array when there are no annotations at all.
    found = sorted(set(raw.annotations.description))
    msg = f"No annotations match event_id {dict(event_id)}; descriptions present: {found}."
    try:
        events, _ = mne.events_from_annotations(raw, event_id=dict(event_id), verbose="error")
    except ValueError:
        raise ValueError(msg) from None
    if events.shape[0] == 0:
        raise ValueError(msg)
    return events


class EEGPipeline:
    """Run a :class:`PreprocessingConfig` on one recording.

    Parameters
    ----------
    config : PreprocessingConfig
        The steps to apply.
    """

    def __init__(self, config: PreprocessingConfig) -> None:
        if not isinstance(config, PreprocessingConfig):
            raise TypeError(f"config must be a PreprocessingConfig, got {type(config).__name__}.")
        self._config = config

    @property
    def config(self) -> PreprocessingConfig:
        return self._config

    def __repr__(self) -> str:
        return f"EEGPipeline({' -> '.join(self._config.ops)})"

    def run(
        self,
        inst: mne.io.BaseRaw | mne.BaseEpochs,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[mne.io.BaseRaw | mne.BaseEpochs, dict[str, Any]]:
        """Apply every step in order.

        Parameters
        ----------
        inst : mne.io.BaseRaw or mne.BaseEpochs
            Input data. Not modified. Continuous input may use any valid
            config; epoched input only configs without continuous steps.
        metadata : mapping or None
            JSON-serializable identifiers stored verbatim in the record, e.g.
            ``{"dataset": "BCI-IV-2a", "subject": "01", "session": "T"}``.

        Returns
        -------
        output : mne.io.BaseRaw or mne.BaseEpochs
            Processed copy (Epochs if the config contains ``make_epochs``).
        record : dict
            JSON-serializable provenance: ``schema_version``, ``config``,
            ``versions``, ``metadata``, ``input`` summary, and ``steps``, one
            entry per step with its ``op``, ``params``, ``output`` summary
            and, where epochs can be lost, ``n_dropped``.

        Raises
        ------
        TypeError
            If ``inst`` is not MNE Raw/Epochs.
        ValueError
            If the config cannot run on this input type, or ``metadata`` is
            not JSON-serializable. Errors raised by a step propagate with a
            note naming the step.
        """
        if isinstance(inst, mne.BaseEpochs):
            start = _EPOCHED
        elif isinstance(inst, mne.io.BaseRaw):
            start = _CONTINUOUS
        else:
            raise TypeError(
                f"Expected mne.io.BaseRaw or mne.Epochs, got {type(inst).__name__}. "
                "Convert arrays with eeg_decoder.io.to_raw first."
            )
        err = _order_error(self._config.ops, start)
        if err is not None:
            raise ValueError(f"This config cannot run on {start} input: {err}")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None.")
        metadata = dict(metadata or {})
        _json_check(metadata, "metadata")

        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "config": self._config.to_dict(),
            "versions": _versions(),
            "metadata": metadata,
            "input": _summary(inst),
            "steps": [],
        }

        current = inst
        for i, step in enumerate(self._config.steps):
            op = step["op"]
            params = {k: _thaw(v) for k, v in step.items() if k != "op"}
            spec = _OPS[op]
            call = {k: v for k, v in params.items() if k not in spec.pipeline_params}
            if op == "baseline_correct":
                call["baseline"] = tuple(call["baseline"])
            entry: dict[str, Any] = {"op": op, "params": params}
            try:
                if op == "make_epochs":
                    events = _extract_events(
                        current, params["event_source"], params.get("stim_channel"),
                        call["event_id"],
                    )
                    n_selected = int(np.isin(events[:, 2], list(call["event_id"].values())).sum())
                    current = make_epochs(current, events, **call)
                    entry["n_events"] = n_selected
                    entry["n_dropped"] = n_selected - len(current)
                elif op == "reject_epochs":
                    n_before = len(current)
                    current = spec.func(current, **call)
                    entry["n_dropped"] = n_before - len(current)
                else:
                    current = spec.func(current, **call)
            except Exception as exc:
                exc.add_note(f"Raised in EEGPipeline step {i} ({op}) with params {params}.")
                raise
            entry["output"] = _summary(current)
            record["steps"].append(entry)

        return current, record
