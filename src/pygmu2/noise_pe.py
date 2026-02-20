"""
NoisePE - generate white, pink, or brown noise.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from pygmu2.source_pe import SourcePE
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class NoiseMode(Enum):
    """Noise color modes."""

    WHITE = "white"
    PINK = "pink"
    BROWN = "brown"


class NoisePE(SourcePE):
    """
    Noise generator.

    Args:
        min_value: Minimum output value (default: -1.0)
        max_value: Maximum output value (default: 1.0)
        seed: Optional seed for deterministic output.
        mode: Noise color (default: WHITE)
    """

    def __init__(
        self,
        min_value: float = -1.0,
        max_value: float = 1.0,
        seed: int | None = None,
        mode: NoiseMode = NoiseMode.WHITE,
    ):
        if max_value < min_value:
            raise ValueError("NoisePE requires max_value >= min_value")

        self._min_value = float(min_value)
        self._max_value = float(max_value)
        self._seed = seed
        self._mode = mode

        self._rng: np.random.Generator | None = None

        # Pink noise filter state (Paul Kellet filter)
        self._pink_b = np.zeros(7, dtype=np.float32)

        # Brown noise state (random walk)
        self._brown_last = 0.0

    @property
    def min_value(self) -> float:
        return self._min_value

    @property
    def max_value(self) -> float:
        return self._max_value

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def mode(self) -> NoiseMode:
        return self._mode

    def inputs(self) -> list:
        return []

    def is_pure(self) -> bool:
        # RNG and filter states are mutable.
        return False

    def channel_count(self) -> int:
        return 1

    def _compute_extent(self) -> Extent:
        return Extent(None, None)

    def _reset_state(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._pink_b[:] = 0.0
        self._brown_last = 0.0

    def _on_start(self) -> None:
        self._reset_state()

    def _on_stop(self) -> None:
        self._rng = None

    def _scale_output(self, x: np.ndarray) -> np.ndarray:
        """Scale from [-1, 1] to [min_value, max_value]."""
        if self._min_value == -1.0 and self._max_value == 1.0:
            return x
        span = self._max_value - self._min_value
        return ((x + 1.0) * 0.5 * span + self._min_value).astype(
            np.float32, copy=False
        )

    def _render_white(self, duration: int) -> np.ndarray:
        # Uniform in [-1, 1]
        return self._rng.uniform(-1.0, 1.0, size=duration).astype(np.float32, copy=False)

    def _render_pink(self, duration: int) -> np.ndarray:
        # Paul Kellet's filter for pink noise
        white = self._render_white(duration)
        out = np.empty(duration, dtype=np.float32)
        b0, b1, b2, b3, b4, b5, b6 = self._pink_b

        for i in range(duration):
            w = white[i]
            b0 = 0.99886 * b0 + w * 0.0555179
            b1 = 0.99332 * b1 + w * 0.0750759
            b2 = 0.96900 * b2 + w * 0.1538520
            b3 = 0.86650 * b3 + w * 0.3104856
            b4 = 0.55000 * b4 + w * 0.5329522
            b5 = -0.7616 * b5 - w * 0.0168980
            pink = b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362
            b6 = w * 0.115926
            out[i] = pink * 0.11  # normalize to roughly [-1, 1]

        self._pink_b[:] = np.array([b0, b1, b2, b3, b4, b5, b6], dtype=np.float32)
        return out

    def _render_brown(self, duration: int) -> np.ndarray:
        # Brown noise: integrate white noise (random walk)
        white = self._render_white(duration)
        out = np.empty(duration, dtype=np.float32)
        last = float(self._brown_last)
        for i in range(duration):
            last = last + white[i] * 0.02
            if last < -1.0:
                last = -1.0
            elif last > 1.0:
                last = 1.0
            out[i] = last
        self._brown_last = last
        return out

    def _render(self, start: int, duration: int) -> Snippet:
        if duration <= 0:
            return Snippet.from_zeros(start, 0, 1)

        if self._mode == NoiseMode.WHITE:
            data = self._render_white(duration)
        elif self._mode == NoiseMode.PINK:
            data = self._render_pink(duration)
        elif self._mode == NoiseMode.BROWN:
            data = self._render_brown(duration)
        else:
            raise ValueError(f"Unknown NoiseMode: {self._mode}")

        data = self._scale_output(data).reshape(-1, 1)
        return Snippet(start, data)

    def __repr__(self) -> str:
        return (
            f"NoisePE(mode={self._mode.value}, "
            f"range=[{self._min_value}, {self._max_value}])"
        )
