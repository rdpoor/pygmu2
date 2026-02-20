# src/pygmu2/random_select_pe.py

from __future__ import annotations

import random
from typing import Sequence, List


from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.trigger_signal import TriggerSignal
from pygmu2.trigger_restart_pe import TriggerRestartPE


class _RandomSelectSourcePE(ProcessingElement):
    """
    A stateful source-wrapper that randomly selects one of N input PEs whenever
    it is reset. Rendering is delegated to the currently-selected input.

    Intended use: wrap this in TriggerRestartPE(trigger, _RandomSelectSourcePE(...))
    so that each positive trigger causes TriggerRestartPE to call reset_state(),
    which rerolls the selection.
    """

    def __init__(
        self,
        inputs: Sequence[ProcessingElement],
        weights: Sequence[float] | None = None,
        seed: int | None = None,
    ):
        if not inputs:
            raise ValueError("_RandomSelectSourcePE requires at least one input")

        if weights is not None and len(weights) != len(inputs):
            raise ValueError("weights must have the same length as inputs")

        self._inputs: List[ProcessingElement] = list(inputs)
        self._weights = list(weights) if weights is not None else None
        self._rng = random.Random(seed)

        self._active: ProcessingElement | None = None

    def inputs(self) -> list[ProcessingElement]:
        # This wrapper depends on all sources.
        return list(self._inputs)

    def is_pure(self) -> bool:
        # Randomness + internal selection state
        return False

    def channel_count(self) -> int | None:
        return self._inputs[0].channel_count()

    def resolve_channel_count(self, input_channel_counts: list[int]) -> int:
        # Require all candidate sources to have same channel count.
        if not input_channel_counts:
            raise ValueError("_RandomSelectSourcePE has no inputs")
        cc0 = input_channel_counts[0]
        for i, cc in enumerate(input_channel_counts[1:], start=1):
            if cc != cc0:
                raise ValueError(
                    f"_RandomSelectSourcePE channel mismatch: input 0 has {cc0}, "
                    f"input {i} has {cc}"
                )
        return cc0

    def _compute_extent(self) -> Extent:
        # Conservative: any source could be chosen at any time; union would be ideal
        # but expensive. Treat as infinite.
        return Extent(None, None)

    def _reset_state(self) -> None:
        # Reroll on reset.
        idxs = list(range(len(self._inputs)))
        idx = self._rng.choices(idxs, weights=self._weights, k=1)[0]
        self._active = self._inputs[idx]

    def _on_start(self) -> None:
        self._reset_state()

    def _on_stop(self) -> None:
        self._active = None

    def _render(self, start: int, duration: int) -> Snippet:
        if self._active is None:
            # No selection yet; choose deterministically now.
            self._reset_state()
        return self._active.render(start, duration)


class RandomSelectPE(ProcessingElement):
    """
    On each positive trigger event, randomly selects one of N input PEs, then
    restarts and renders that selection from local time 0.

    Implemented as:
        TriggerRestartPE(trigger, _RandomSelectSourcePE(inputs, ...))

    Args:
        trigger: TriggerSignal; trigger > 0 indicates a (rising-edge) event.
        inputs: candidate audio sources (must match channel_count).
        weights: optional selection weights (random.choices semantics).
        seed: optional RNG seed for deterministic selection.
    """

    def __init__(
        self,
        trigger: TriggerSignal,
        inputs: Sequence[ProcessingElement],
        weights: Sequence[float] | None = None,
        seed: int | None = None,
    ):
        if not inputs:
            raise ValueError("RandomSelectPE requires at least one input")

        self._trigger = trigger
        self._sources = list(inputs)

        self._selector = _RandomSelectSourcePE(self._sources, weights=weights, seed=seed)
        self._impl = TriggerRestartPE(self._trigger, self._selector)

    def inputs(self) -> list[ProcessingElement]:
        # Expose the true dependency set for graph validation/traversal.
        return [self._trigger] + self._sources

    def is_pure(self) -> bool:
        return False  # selection + trigger restart are stateful

    def channel_count(self) -> int | None:
        return self._selector.channel_count()

    def resolve_channel_count(self, input_channel_counts: list[int]) -> int:
        # input_channel_counts includes trigger at index 0
        if len(input_channel_counts) < 2:
            raise ValueError("RandomSelectPE has no audio inputs")
        audio_counts = input_channel_counts[1:]
        cc0 = audio_counts[0]
        for i, cc in enumerate(audio_counts[1:], start=2):
            if cc != cc0:
                raise ValueError(
                    f"RandomSelectPE channel mismatch: input 1 has {cc0}, input {i} has {cc}"
                )
        return cc0

    def _compute_extent(self) -> Extent:
        # Trigger-driven: silence until first event; after that depends on sources.
        # Conservative choice is trigger extent.
        return self._trigger.extent()

    def _reset_state(self) -> None:
        # Reset both layers.
        self._selector.reset_state()
        self._impl.reset_state()

    def _on_start(self) -> None:
        self._selector.on_start()
        self._impl.on_start()

    def _on_stop(self) -> None:
        self._impl.on_stop()
        self._selector.on_stop()

    def _render(self, start: int, duration: int) -> Snippet:
        # Delegate to the composed implementation.
        return self._impl.render(start, duration)
