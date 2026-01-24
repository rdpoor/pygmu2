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
    
    Takes a list of (PE, start_time) pairs and plays them sequentially.
    Each PE is delayed by its start_time. When overlap=False, each PE
    is cropped to prevent overlap with the next PE.
    
    The sequence is automatically sorted by start_time.
    
    Args:
        sequence: List of (PE, start_time) tuples where start_time is in samples
        overlap: If False, crop each PE to prevent overlap (default: False)
    
    Example:
        # Non-overlapping sequence
        pe1 = SinePE(frequency=440.0)
        pe2 = SinePE(frequency=550.0)
        sequence = [
            (pe1, 0),      # Starts at sample 0
            (pe2, 44100),  # Starts at sample 44100 (1 second at 44.1kHz)
        ]
        seq = SequencePE(sequence, overlap=False)
        
        # Overlapping sequence (all play simultaneously after delays)
        seq_overlap = SequencePE(sequence, overlap=True)
    """
    
    def __init__(
        self,
        sequence: list[tuple[ProcessingElement, int]],
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
        
        for i, (pe, start_time) in enumerate(sorted_sequence):
            # Apply cropping if overlap=False
            if not overlap and i < len(sorted_sequence) - 1:
                # Not the last item: crop PE to [0, next_start_time - start_time)
                # in its own timeline, so it plays for (next_start_time - start_time) samples
                next_start_time = sorted_sequence[i + 1][1]
                duration = next_start_time - start_time
                if duration > 0:
                    cropped_pe = CropPE(pe, Extent(0, duration))
                else:
                    # Zero or negative duration: don't crop (PEs with same start_time will mix)
                    cropped_pe = pe
            elif not overlap:
                # Last item: no cropping (let it play to its natural end)
                cropped_pe = pe
            else:
                # overlap=True: no cropping
                cropped_pe = pe
            
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
    def sequence(self) -> list[tuple[ProcessingElement, int]]:
        """The sequence of (PE, start_time) pairs (sorted by start_time)."""
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
        return [pe for pe, _ in self._sequence]
    
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
