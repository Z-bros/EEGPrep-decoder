"""Core configuration shared by all EEG_Decoder operations.

The configuration is deliberately tiny. It holds only settings that affect
*every* stochastic or numerical operation, so that an experiment can record
them and be reproduced. Operation-specific settings (filter cutoffs, epoch
windows, ...) will live in their own config objects in later phases.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

_ALLOWED_DTYPES = ("float32", "float64")


@dataclass(frozen=True)
class EEGConfig:
    """Global, immutable settings for reproducible EEG processing.

    Parameters
    ----------
    seed : int, default=0
        Seed for every stochastic operation. There is no hidden global RNG:
        functions that need randomness must take a seed or a
        ``numpy.random.Generator`` derived from this value.
    float_dtype : {"float32", "float64"}, default="float64"
        Floating-point precision used when this package creates new arrays.
        Inputs are never silently cast; this only governs outputs we allocate.

    Raises
    ------
    ValueError
        If ``seed`` is negative or ``float_dtype`` is not supported.
    TypeError
        If ``seed`` is not an ``int`` (``bool`` is rejected explicitly).

    Notes
    -----
    The dataclass is frozen so a config cannot change after an experiment
    starts; use :meth:`replace` to derive a modified copy.
    """

    seed: int = 0
    float_dtype: str = "float64"

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError(f"seed must be an int, got {type(self.seed).__name__}.")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}.")
        if self.float_dtype not in _ALLOWED_DTYPES:
            raise ValueError(
                f"float_dtype must be one of {_ALLOWED_DTYPES}, got {self.float_dtype!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable dict (for experiment records)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EEGConfig:
        """Build a config from :meth:`to_dict` output.

        Unknown keys raise ``TypeError`` rather than being ignored, so a typo
        in a saved experiment record cannot silently fall back to a default.
        """
        return cls(**data)

    def replace(self, **changes: Any) -> EEGConfig:
        """Return a copy with the given fields changed (re-validated)."""
        return type(self)(**{**self.to_dict(), **changes})
