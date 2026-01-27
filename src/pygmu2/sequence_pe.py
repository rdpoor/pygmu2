"""
SequencePE - sequences multiple PEs in time.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

from typing import Optional

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
    
    Takes a list of (PE, start_time) pairs or (PE, start_time, duration) triples.
    Each PE is delayed by its start_time. When overlap=False, each PE is cropped
    to prevent overlap with the next PE (or to the specified duration).
    
    The sequence is automatically sorted by start_time.
    
    Args:
        sequence: List of tuples, either:
            - (PE, start_time) - duration inferred from next item's start_time
            - (PE, start_time, duration) - explicit duration in samples
            Both formats can be mixed in the same sequence.
        overlap: If False, crop each PE to prevent overlap (default: False)
    
    Example:
        # Non-overlapping sequence with inferred durations
        pe1 = SinePE(frequency=440.0)
        pe2 = SinePE(frequency=550.0)
        sequence = [
            (pe1, 0),      # Starts at sample 0, plays until pe2 starts
            (pe2, 44100),  # Starts at sample 44100 (1 second at 44.1kHz)
        ]
        seq = SequencePE(sequence, overlap=False)
        
        # Overlapping sequence with explicit durations
        sequence_timed = [
            (pe1, 0, 88200),       # Starts at 0, plays for 2 seconds
            (pe2, 44100, 44100),   # Starts at 1s, plays for 1 second (overlaps pe1)
        ]
        seq_overlap = SequencePE(sequence_timed, overlap=True)
        
        # Mixed format (both 2-tuples and 3-tuples)
        sequence_mixed = [
            (pe1, 0, 44100),  # Explicit duration
            (pe2, 44100),     # Inferred duration
        ]
        seq_mixed = SequencePE(sequence_mixed, overlap=False)
    """
    
    def __init__(
        self,
        sequence: list[tuple[ProcessingElement, int] | tuple[ProcessingElement, int, int]],
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
        
        # Sort by start_time
        sorted_sequence = sorted(sequence, key=lambda x: x[1])
        self._sequence = sorted_sequence
        
        # Build the processing chain
        processed_pes = []
        
        for i, item in enumerate(sorted_sequence):
            # Parse item - can be (PE, start_time) or (PE, start_time, duration)
            if len(item) == 3:
                pe, start_time, explicit_duration = item
            else:
                pe, start_time = item
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
                elif i < len(sorted_sequence) - 1:
                    # Infer duration from next item's start_time
                    next_start_time = sorted_sequence[i + 1][1]
                    duration = next_start_time - start_time
                    if duration > 0:
                        cropped_pe = CropPE(pe, Extent(0, duration))
                # else: last item with no explicit duration - no cropping
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
    def sequence(self) -> list[tuple[ProcessingElement, int] | tuple[ProcessingElement, int, int]]:
        """The sequence of items (sorted by start_time).
        
        Each item is either (PE, start_time) or (PE, start_time, duration).
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
        return [item[0] for item in self._sequence]
    
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
