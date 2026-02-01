import math
from array import array
from collections.abc import MutableSequence, Sequence
import numpy as np
import itertools

# Module-level constants for hot paths (avoids repeated log/calls in voice/envelope).
NON_AUDIBLE = 1.0e-3
LOG_NON_AUDIBLE = math.log(NON_AUDIBLE)
HALF_PI = math.pi / 2


def create_buffer(length: int) -> MutableSequence[float]:
    return array("d", itertools.repeat(0, length))


def create_buffer_numpy(length: int) -> np.ndarray:
    """Create a zeroed float64 buffer as a NumPy array for vectorized hot paths."""
    return np.zeros(length, dtype=np.float64)


class SoundFontMath:
    @staticmethod
    def half_pi() -> float:
        return math.pi / 2

    @staticmethod
    def non_audible() -> float:
        return 1.0e-3

    @staticmethod
    def log_non_audible() -> float:
        return math.log(1.0e-3)

    @staticmethod
    def timecents_to_seconds(x: float) -> float:
        return math.pow(2.0, (1.0 / 1200.0) * x)

    @staticmethod
    def cents_to_hertz(x: float) -> float:
        return 8.176 * math.pow(2.0, (1.0 / 1200.0) * x)

    @staticmethod
    def cents_to_multiplying_factor(x: float) -> float:
        return math.pow(2.0, (1.0 / 1200.0) * x)

    @staticmethod
    def decibels_to_linear(x: float) -> float:
        return math.pow(10.0, 0.05 * x)

    @staticmethod
    def linear_to_decibels(x: float) -> float:
        return 20.0 * math.log10(x)

    @staticmethod
    def key_number_to_multiplying_factor(cents: int, key: int) -> float:
        return SoundFontMath.timecents_to_seconds(cents * (60 - key))

    @staticmethod
    def exp_cutoff(x: float) -> float:
        return 0.0 if x < LOG_NON_AUDIBLE else math.exp(x)

    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(max_val, value))


class ArrayMath:
    @staticmethod
    def multiply_add(
        a: float, x: Sequence[float], destination: MutableSequence[float]
    ) -> None:
        if isinstance(destination, np.ndarray) and isinstance(x, np.ndarray):
            destination += a * x
        else:
            for i in range(len(destination)):
                destination[i] += a * x[i]

    @staticmethod
    def multiply_add_slope(
        a: float,
        step: float,
        x: Sequence[float],
        destination: MutableSequence[float],
    ) -> None:
        if isinstance(destination, np.ndarray) and isinstance(x, np.ndarray):
            n = len(destination)
            ramp = a + step * np.arange(n, dtype=np.float64)
            destination += ramp * x
        else:
            for i in range(len(destination)):
                destination[i] += a * x[i]
                a += step
