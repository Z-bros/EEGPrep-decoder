"""A small labelled-array container for feature outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import mne
import numpy as np


def _label(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):g}"
    return str(value)


@dataclass(frozen=True, eq=False)
class FeatureArray:
    """Feature values with named axes, aligned row-for-row with the epochs.

    Attributes
    ----------
    data : ndarray
        Read-only array whose first axis is ``"epoch"``.
    dims : tuple of str
        Axis names, e.g. ``("epoch", "channel", "band")``.
    coords : mapping of str to tuple
        Labels for every axis except ``"epoch"`` (channel names, band
        names, frequencies in Hz, times in s, ...).
    units : str
        Units of ``data``, e.g. ``"V^2/Hz"``, ``"V^2"``, ``"dB re 1 uV^2"``.
    events : ndarray, shape (n_epochs, 3)
        Read-only copy of ``epochs.events``: row ``i`` of ``data`` belongs to
        event ``events[i]`` (use ``events[:, 2]`` with ``event_id`` as labels).
    event_id : mapping of str to int
        Condition names to event codes.
    """

    data: np.ndarray
    dims: tuple[str, ...]
    coords: Mapping[str, tuple]
    units: str
    events: np.ndarray
    event_id: Mapping[str, int]

    @classmethod
    def _from_epochs(
        cls,
        epochs: mne.BaseEpochs,
        data: np.ndarray,
        dims: Sequence[str],
        coords: Mapping[str, Sequence],
        units: str,
    ) -> FeatureArray:
        data = np.array(data, copy=True)
        data.setflags(write=False)
        events = np.array(epochs.events, copy=True)
        events.setflags(write=False)
        dims = tuple(dims)
        if dims[0] != "epoch" or data.ndim != len(dims):
            raise AssertionError("internal: dims must start with 'epoch' and match data.ndim")
        for axis, dim in enumerate(dims[1:], start=1):
            if len(coords[dim]) != data.shape[axis]:
                raise AssertionError(f"internal: coords[{dim!r}] does not match data axis {axis}")
        return cls(
            data=data,
            dims=dims,
            coords=MappingProxyType({d: tuple(coords[d]) for d in dims[1:]}),
            units=units,
            events=events,
            event_id=MappingProxyType(dict(epochs.event_id)),
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def flatten(self, sep: str = ":") -> tuple[np.ndarray, list[str]]:
        """Return ``(X, names)`` with ``X`` of shape ``(n_epochs, n_features)``.

        Columns are ordered by the remaining axes in ``dims`` order (e.g.
        channel-major for ``("epoch", "channel", "band")``), and each name
        joins the labels, e.g. ``"C3:alpha"``. ``X`` is a writable copy.
        """
        rest = self.dims[1:]
        grids = np.meshgrid(*[np.arange(len(self.coords[d])) for d in rest], indexing="ij")
        names = [
            sep.join(_label(self.coords[d][int(g.flat[k])]) for d, g in zip(rest, grids))
            for k in range(grids[0].size)
        ] if rest else []
        return self.data.reshape(self.data.shape[0], -1).copy(), names

    def __repr__(self) -> str:
        sizes = ", ".join(f"{d}: {n}" for d, n in zip(self.dims, self.data.shape))
        return f"FeatureArray({sizes}; units={self.units!r})"
