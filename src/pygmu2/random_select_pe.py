"""
RandomSelectPE - choose one input at start and render it on trigger.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

from typing import List, Optional, Sequence
import random

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.logger import get_logger
from pygmu2.trigger_pe import TriggerPE, TriggerMode

logger = get_logger(__name__)


class RandomSelectPE(ProcessingElement):
    """
    Randomly select one input at start and render it on trigger.

    Each time rendering starts, a single input is chosen using weighted
    probabilities (if provided). The chosen input is then wrapped in a TriggerPE
    and rendered from its t=0 when the trigger goes positive.

    Args:
        trigger: Control PE (trigger signal, mono channel 0).
        inputs: List of candidate input PEs to choose from. You may also pass
            inputs as positional arguments: RandomSelectPE(trigger, pe1, pe2, ...).
        weights: Optional list of weights (same length as inputs), using
            random.choices semantics.
        seed: Optional seed for deterministic selection.
        trigger_mode: Trigger behavior (default: RETRIGGER).
    """

    def __init__(
        self,
        trigger: ProcessingElement,
        inputs: List[ProcessingElement] | ProcessingElement | None = None,
        *more_inputs: ProcessingElement,
        weights: Optional[Sequence[float]] = None,
        seed: Optional[int] = None,
        trigger_mode: TriggerMode = TriggerMode.RETRIGGER,
    ):
        if more_inputs:
            if inputs is None or isinstance(inputs, (list, tuple)):
                raise ValueError(
                    "RandomSelectPE: pass either a single inputs list/tuple or positional inputs"
                )
            inputs_list = [inputs, *more_inputs]
        else:
            if inputs is None:
                inputs_list = []
            elif isinstance(inputs, (list, tuple)):
                inputs_list = list(inputs)
            else:
                inputs_list = [inputs]

        if not inputs_list:
            raise ValueError("RandomSelectPE requires at least one input")
        if weights is not None and len(weights) != len(inputs_list):
            raise ValueError("weights must have the same length as inputs")

        self._trigger = trigger
        self._inputs = list(inputs_list)
        self._weights = list(weights) if weights is not None else None
        self._rng = random.Random(seed)
        self._trigger_mode = trigger_mode

        self._selector = _RandomSelectSourcePE(
            inputs=self._inputs,
            weights=self._weights,
            rng=self._rng,
        )
        self._trigger_pe = TriggerPE(
            self._selector,
            self._trigger,
            trigger_mode=self._trigger_mode,
        )

    def inputs(self) -> list[ProcessingElement]:
        return [self._trigger] + self._inputs

    def is_pure(self) -> bool:
        # Selection is stateful and triggered rendering is stateful.
        return False

    def channel_count(self) -> Optional[int]:
        return self._inputs[0].channel_count()

    def _compute_extent(self) -> Extent:
        # Extent determined by trigger source.
        return self._trigger.extent()

    def required_input_channels(self) -> Optional[int]:
        return None

    def resolve_channel_count(self, input_channel_counts: list[int]) -> int:
        if not input_channel_counts:
            raise ValueError("RandomSelectPE has no inputs")
        # input_channel_counts includes trigger; inputs start at index 1
        input_counts = input_channel_counts[1:]
        first = input_counts[0]
        for i, count in enumerate(input_counts[1:], start=2):
            if count != first:
                raise ValueError(
                    f"RandomSelectPE input channel mismatch: input 1 has {first} channels, "
                    f"input {i} has {count} channels"
                )
        return first

    def _reset_state(self) -> None:
        self._selector.reset_state()
        self._trigger_pe.reset_state()

    def _on_start(self) -> None:
        # Reroll selection on trigger by resetting selector state.
        self._reset_state()

    def _on_stop(self) -> None:
        self._reset_state()

    def _render(self, start: int, duration: int) -> Snippet:
        return self._trigger_pe.render(start, duration)


class _RandomSelectSourcePE(ProcessingElement):
    """
    Selects one input on reset_state() and renders only that input.
    """

    def __init__(
        self,
        inputs: List[ProcessingElement],
        weights: Optional[Sequence[float]],
        rng: random.Random,
    ):
        self._inputs = list(inputs)
        self._weights = list(weights) if weights is not None else None
        self._rng = rng
        self._selected_index: Optional[int] = None
        self._selected_source: Optional[ProcessingElement] = None

    def inputs(self) -> list[ProcessingElement]:
        return self._inputs

    def is_pure(self) -> bool:
        return False

    def channel_count(self) -> Optional[int]:
        return self._inputs[0].channel_count()

    def _compute_extent(self) -> Extent:
        # Union of inputs (conservative).
        result = self._inputs[0].extent()
        for inp in self._inputs[1:]:
            result = result.union(inp.extent())
        return result

    def _reset_state(self) -> None:
        indices = list(range(len(self._inputs)))
        self._selected_index = self._rng.choices(indices, weights=self._weights, k=1)[0]
        self._selected_source = self._inputs[self._selected_index]
        logger.debug(
            "RandomSelectPE selected input %d (%s)",
            self._selected_index,
            self._selected_source.__class__.__name__,
        )
        self._selected_source.reset_state()

    def _on_start(self) -> None:
        self._reset_state()

    def _on_stop(self) -> None:
        self._selected_index = None
        self._selected_source = None

    def _render(self, start: int, duration: int) -> Snippet:
        if self._selected_source is None:
            self._reset_state()
        return self._selected_source.render(start, duration)
