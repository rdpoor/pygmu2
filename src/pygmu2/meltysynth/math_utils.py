import math
from array import array
from collections.abc import MutableSequence, Sequence

import itertools


def create_buffer(length: int) -> MutableSequence[float]:
    return array("d", itertools.repeat(0, length))


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
        return (
            0.0
            if x < SoundFontMath.log_non_audible()
            else math.exp(x)
        )

    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(max_val, value))


class ArrayMath:
    @staticmethod
    def multiply_add(
        a: float, x: Sequence[float], destination: MutableSequence[float]
    ) -> None:
        for i in range(len(destination)):
            destination[i] += a * x[i]

    @staticmethod
    def multiply_add_slope(
        a: float,
        step: float,
        x: Sequence[float],
        destination: MutableSequence[float],
    ) -> None:
        for i in range(len(destination)):
            destination[i] += a * x[i]
            a += step
