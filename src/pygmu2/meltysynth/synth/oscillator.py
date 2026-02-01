from __future__ import annotations

import math
from collections.abc import MutableSequence, Sequence
from typing import TYPE_CHECKING

from pygmu2.meltysynth.model.types import LoopMode

if TYPE_CHECKING:
    from pygmu2.meltysynth.synth.synthesizer import Synthesizer


class Oscillator:
    def __init__(self, synthesizer: Synthesizer) -> None:
        self._synthesizer = synthesizer

    def start(
        self,
        data: Sequence[float],
        loop_mode: LoopMode,
        sample_rate: int,
        start: int,
        end: int,
        start_loop: int,
        end_loop: int,
        root_key: int,
        coarse_tune: int,
        fine_tune: int,
        scale_tuning: int,
    ) -> None:
        self._data = data
        self._loop_mode = loop_mode
        self._sample_rate = sample_rate
        self._start = start
        self._end = end
        self._start_loop = start_loop
        self._end_loop = end_loop
        self._root_key = root_key

        self._tune = coarse_tune + 0.01 * fine_tune
        self._pitch_change_scale = 0.01 * scale_tuning
        self._sample_rate_ratio = sample_rate / self._synthesizer.sample_rate

        if loop_mode == LoopMode.NO_LOOP:
            self._looping = False
        else:
            self._looping = True

        self._position = start

    def release(self) -> None:
        if self._loop_mode == LoopMode.LOOP_UNTIL_NOTE_OFF:
            self._looping = False

    def process(self, block: MutableSequence[float], pitch: float) -> bool:
        pitch_change = (
            self._pitch_change_scale * (pitch - self._root_key) + self._tune
        )
        pitch_ratio = self._sample_rate_ratio * math.pow(
            2.0, pitch_change / 12.0
        )
        return self.fill_block(block, pitch_ratio)

    def fill_block(
        self, block: MutableSequence[float], pitch_ratio: float
    ) -> bool:
        if self._looping:
            return self.fill_block_continuous(block, pitch_ratio)
        else:
            return self.fill_block_no_loop(block, pitch_ratio)

    def fill_block_no_loop(
        self, block: MutableSequence[float], pitch_ratio: float
    ) -> bool:
        for t in range(len(block)):
            index = int(self._position)

            if index >= self._end:
                if t > 0:
                    for u in range(t, len(block)):
                        block[u] = 0
                    return True
                else:
                    return False

            x1 = self._data[index]
            x2 = self._data[index + 1]
            a = self._position - index
            block[t] = x1 + a * (x2 - x1)

            self._position += pitch_ratio

        return True

    def fill_block_continuous(
        self, block: MutableSequence[float], pitch_ratio: float
    ) -> bool:
        end_loop_position = float(self._end_loop)
        loop_length = self._end_loop - self._start_loop

        for t in range(len(block)):
            if self._position >= end_loop_position:
                self._position -= loop_length

            index1 = int(self._position)
            index2 = index1 + 1

            if index2 >= self._end_loop:
                index2 -= loop_length

            x1 = self._data[index1]
            x2 = self._data[index2]
            a = self._position - index1
            block[t] = x1 + a * (x2 - x1)

            self._position += pitch_ratio

        return True
