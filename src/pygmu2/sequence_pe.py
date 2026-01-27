"""
SequencePE - sequences multiple PEs in time.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

from typing import Optional, Union

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.delay_pe import DelayPE
from pygmu2.crop_pe import CropPE
from pygmu2.mix_pe import MixPE
from pygmu2.constant_pe import ConstantPE
from pygmu2.config import handle_error


class SequencePE(ProcessingElement):
    """
    A ProcessingElement that sequences multiple PEs in time.
    
    Takes a list of (item, start_time) pairs or (item, start_time, duration) triples.
    Each item may be either a ProcessingElement or a scalar (float/int).
    Each PE is delayed by its start_time. When overlap=False, each PE is cropped
    to prevent overlap with the next PE (or to the specified duration).
    
    The sequence is automatically sorted by start_time.
    
    Args:
        sequence: List of tuples, either:
            - (item, start_time) - duration inferred from next item's start_time
            - (item, start_time, duration) - explicit duration in samples
            where item is either a ProcessingElement or a scalar (float/int).
            Both formats can be mixed in the same sequence.
        channels: Output channel count for scalar items. If one or more items are
                  ProcessingElements, this must match their channel count.
                  If all items are scalars, defaults to 1.
        overlap: If False, crop each PE to prevent overlap (default: False)
    
    Example:
        # Non-overlapping sequence with inferred durations
        pe1 = SinePE(frequency=440.0)
        pe2 = SinePE(frequency=550.0)
        sequence = [
            (pe1, 0),      # Starts at sample 0, plays until pe2 starts
            (pe2, 44100),  # Starts at sample 44100 (1 second at 44.1kHz)
        ]
        seq_stream = SequencePE(sequence, overlap=False)
        
        # Overlapping sequence with explicit durations
        sequence_timed = [
            (pe1, 0, 88200),       # Starts at 0, plays for 2 seconds
            (pe2, 44100, 44100),   # Starts at 1s, plays for 1 second (overlaps pe1)
        ]
        seq_overlap = SequencePE(sequence_timed, overlap=True)
        
        # A scalar "step sequence" (common for control signals)
        values = [
            (0.0, 0),     # 0.0 starting at t=0
            (0.5, 100),   # 0.5 starting at t=100 samples
            (1.0, 200),   # 1.0 starting at t=200 samples
        ]
        steps_stream = SequencePE(values, overlap=False)
        
        # Mixed format (both 2-tuples and 3-tuples)
        sequence_mixed = [
            (pe1, 0, 44100),  # Explicit duration
            (pe2, 44100),     # Inferred duration
        ]
        seq_mixed = SequencePE(sequence_mixed, overlap=False)
    """
    
    def __init__(
        self,
        sequence: list[tuple[Union[ProcessingElement, float, int], int] | tuple[Union[ProcessingElement, float, int], int, int]],
        channels: Optional[int] = None,
        overlap: bool = False,
    ):
        if not sequence:
            # Empty sequence: handle based on error mode
            if handle_error("SequencePE: empty sequence provided", fatal=False):
                # LENIENT mode: use silence
                self._mix_pe = ConstantPE(0.0, channels=1)
            else:
                # STRICT mode: exception was raised, we won't get here
                self._mix_pe = None
            self._sequence = []
            self._overlap = overlap
            return
        
        self._overlap = overlap

        # Determine channel count.
        #
        # Rule:
        # - If one or more items are PEs, the first PE with a known channel_count()
        #   sets the base channel count. Any subsequent PEs with known channel counts
        #   must match.
        # - If all items are scalars, default to 1 channel, unless channels= is given.
        #
        # Note: We use the input list order (not time-sorted order) to choose the
        # "first PE" so behavior is stable even when start_time ties change sorting.
        base_channels: Optional[int] = None
        saw_pe = False
        saw_unknown_pe_channels = False

        # Extract item (first element) from tuples, handling both 2-tuple and 3-tuple formats
        for seq_item in sequence:
            item = seq_item[0]  # First element is always the PE or scalar
            if isinstance(item, ProcessingElement):
                saw_pe = True
                ch = item.channel_count()
                if ch is None:
                    saw_unknown_pe_channels = True
                    continue
                base_channels = int(ch)
                break

        if base_channels is None:
            # No PE with known channel count in the sequence.
            if channels is not None:
                base_channels = int(channels)
            else:
                base_channels = 1
            # If we saw PEs but none could report a channel count, require explicit channels.
            if saw_pe and saw_unknown_pe_channels and channels is None:
                raise ValueError(
                    "SequencePE: channels must be provided when the sequence contains "
                    "ProcessingElements with unknown channel count"
                )

        if base_channels <= 0:
            raise ValueError(f"SequencePE: channels must be >= 1 (got {base_channels})")

        if channels is not None and int(channels) != base_channels:
            raise ValueError(
                f"SequencePE: channels must match PE channel count (got channels={channels}, "
                f"expected {base_channels})"
            )
        self._channels = base_channels
        
        # Sort by start_time (always the second element)
        sorted_sequence = sorted(sequence, key=lambda x: x[1])
        
        # Normalize items: convert scalars to PEs, handle both 2-tuple and 3-tuple formats
        normalized: list[tuple[ProcessingElement, int] | tuple[ProcessingElement, int, int]] = []
        for seq_item in sorted_sequence:
            item = seq_item[0]  # PE or scalar
            start_time = seq_item[1]  # start time
            explicit_duration = seq_item[2] if len(seq_item) == 3 else None
            
            if isinstance(item, ProcessingElement):
                pe = item
            else:
                # Convert scalar to ConstantPE (gated to be 0 for local time < 0)
                pe = CropPE(ConstantPE(float(item), channels=self._channels), Extent(0, None))
            
            # Preserve format (2-tuple or 3-tuple)
            if explicit_duration is not None:
                normalized.append((pe, int(start_time), int(explicit_duration)))
            else:
                normalized.append((pe, int(start_time)))

        self._sequence = normalized

        # Validate PE channel counts against base_channels when known.
        for seq_item in self._sequence:
            pe = seq_item[0]
            ch = pe.channel_count()
            if ch is None:
                continue
            if int(ch) != self._channels:
                raise ValueError(
                    f"SequencePE input channel mismatch: expected {self._channels} channels, got {ch}"
                )
        
        # Build the processing chain
        processed_pes = []
        
        for i, seq_item in enumerate(self._sequence):
            # Parse item - can be (PE, start_time) or (PE, start_time, duration)
            if len(seq_item) == 3:
                pe, start_time, explicit_duration = seq_item
            else:
                pe, start_time = seq_item
                explicit_duration = None
            
            # Determine if we should crop and what duration to use
            cropped_pe = pe
            
            if not overlap:
                # Need to determine crop duration
                if explicit_duration is not None:
                    # Use explicit duration from 3-tuple
                    duration = explicit_duration
                    if duration > 0:
                        cropped_pe = CropPE(pe, Extent(0, duration))
                elif i < len(self._sequence) - 1:
                    # Infer duration from next item's start_time
                    next_start_time = self._sequence[i + 1][1]
                    duration = next_start_time - start_time
                    if duration > 0:
                        cropped_pe = CropPE(pe, Extent(0, duration))
                    # else: zero or negative duration - don't crop (PEs with same start_time will mix)
                # else: last item with no explicit duration - no cropping (let it play to its natural end)
            elif explicit_duration is not None:
                # overlap=True but explicit duration provided - still crop to duration
                if explicit_duration > 0:
                    cropped_pe = CropPE(pe, Extent(0, explicit_duration))
            
            # Apply delay
            delayed_pe = DelayPE(cropped_pe, delay=start_time)
            processed_pes.append(delayed_pe)
        
        # Mix all processed PEs
        if len(processed_pes) == 1:
            # Single item: no need for MixPE
            self._mix_pe = processed_pes[0]
        else:
            # Multiple items: use MixPE
            self._mix_pe = MixPE(*processed_pes)
    
    @property
    def sequence(self) -> list[tuple[Union[ProcessingElement, float, int], int] | tuple[Union[ProcessingElement, float, int], int, int]]:
        """The sequence of items (sorted by start_time).
        
        Each item is either (PE or scalar, start_time) or (PE or scalar, start_time, duration).
        Note: Scalars are internally converted to ConstantPE instances.
        """
        return self._sequence
    
    @property
    def overlap(self) -> bool:
        """Whether overlapping is allowed."""
        return self._overlap
    
    def inputs(self) -> list[ProcessingElement]:
        """Return all PEs from the sequence."""
        if not self._sequence:
            # Empty sequence: ConstantPE has no inputs
            return []
        # First element is always the PE (scalars are already converted to ConstantPE)
        return [seq_item[0] for seq_item in self._sequence]
    
    def is_pure(self) -> bool:
        """SequencePE is pure - it's a stateless composition."""
        return True
    
    def channel_count(self) -> Optional[int]:
        """Return channel count from internal MixPE (or ConstantPE for empty)."""
        if self._mix_pe is not None:
            return self._mix_pe.channel_count()
        return None
    
    def _compute_extent(self) -> Extent:
        """Return the union of all delayed (and optionally cropped) PEs."""
        if not self._sequence:
            # Empty sequence: infinite extent (ConstantPE)
            return Extent(None, None)
        
        # Get extent from internal MixPE
        return self._mix_pe.extent()
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Render the sequenced audio.
        
        Args:
            start: Starting sample index
            duration: Number of samples to generate (> 0)
        
        Returns:
            Snippet containing the sequenced audio
        """
        if self._mix_pe is not None:
            return self._mix_pe.render(start, duration)
        
        # Should not reach here (empty sequence in STRICT mode raises)
        # But handle gracefully just in case
        return Snippet.from_zeros(start, duration, 1)
    
    def __repr__(self) -> str:
        count = len(self._sequence)
        overlap_str = "overlap=True" if self._overlap else "overlap=False"
        return f"SequencePE({count} items, {overlap_str})"
