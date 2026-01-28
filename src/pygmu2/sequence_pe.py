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
    
    Takes a list of (item, start_time) pairs and plays them sequentially.
    Each item may be either a ProcessingElement or a scalar (float/int).
    Each PE is delayed by its start_time and mixed together.
    
    Behavior:
    - All inputs are delayed to their start_time and mixed (overlap is always allowed)
    - Scalars are automatically converted to ConstantPE and cropped to the next item's
      start time (if there is a next item). The last scalar has infinite extent.
    - For non-overlapping behavior, callers must pre-crop their PEs to the desired
      duration before passing them to SequencePE.
    
    The extent of SequencePE is determined by the start of the first input and
    the end of the last input.
    
    The sequence is automatically sorted by start_time.
    
    Args:
        sequence: List of (item, start_time) tuples where start_time is in samples
                  and item is either a ProcessingElement or a scalar.
        channels: Output channel count for scalar items. If one or more items are
                  ProcessingElements, this must match their channel count.
                  If all items are scalars, defaults to 1.
    
    Example:
        # Sequence of PEs (caller must pre-crop if non-overlapping behavior desired)
        pe1_stream = CropPE(SinePE(frequency=440.0), Extent(0, 44100))
        pe2_stream = CropPE(SinePE(frequency=550.0), Extent(0, 44100))
        sequence = [
            (pe1_stream, 0),      # Starts at sample 0
            (pe2_stream, 44100),  # Starts at sample 44100 (1 second at 44.1kHz)
        ]
        seq_stream = SequencePE(sequence)
        
        # A scalar "step sequence" (auto-cropped to next item's start)
        values = [
            (0.0, 0),     # 0.0 from t=0 to t=100 (cropped to next start)
            (0.5, 100),   # 0.5 from t=100 to t=200 (cropped to next start)
            (1.0, 200),   # 1.0 from t=200 onward (infinite, last item)
        ]
        steps_stream = SequencePE(values)
    """
    
    def __init__(
        self,
        sequence: list[tuple[Union[ProcessingElement, float, int], int]],
        channels: Optional[int] = None,
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
            return

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

        for item, _start_time in sequence:
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
        
        # Sort by start_time
        sorted_sequence = sorted(sequence, key=lambda x: x[1])
        # Normalize scalars to PEs with auto-cropping to next item's start
        normalized: list[tuple[ProcessingElement, int]] = []
        for i, (item, start_time) in enumerate(sorted_sequence):
            if isinstance(item, ProcessingElement):
                pe = item
            else:
                # Scalar: convert to ConstantPE and auto-crop to next item's start
                # (if there is a next item)
                if i < len(sorted_sequence) - 1:
                    next_start = sorted_sequence[i + 1][1]
                    duration = next_start - start_time
                    if duration > 0:
                        pe = CropPE(ConstantPE(float(item), channels=self._channels), Extent(0, duration))
                    else:
                        # Zero or negative duration - use infinite extent (will mix with next)
                        pe = CropPE(ConstantPE(float(item), channels=self._channels), Extent(0, None))
                else:
                    # Last item: infinite extent
                    pe = CropPE(ConstantPE(float(item), channels=self._channels), Extent(0, None))
            normalized.append((pe, int(start_time)))

        self._sequence = normalized

        # Validate PE channel counts against base_channels when known.
        for pe, _start_time in self._sequence:
            ch = pe.channel_count()
            if ch is None:
                continue
            if int(ch) != self._channels:
                raise ValueError(
                    f"SequencePE input channel mismatch: expected {self._channels} channels, got {ch}"
                )
        
        # Build the processing chain: delay and mix all inputs
        processed_pes = []
        for pe, start_time in self._sequence:
            delayed_pe = DelayPE(pe, delay=start_time)
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
    
    
    def inputs(self) -> list[ProcessingElement]:
        """Return the internal MixPE (or DelayPE for single item)."""
        if self._mix_pe is not None:
            return [self._mix_pe]
        return []
    
    def is_pure(self) -> bool:
        """SequencePE is pure - it's a stateless composition."""
        return True
    
    def channel_count(self) -> Optional[int]:
        """Return channel count from internal MixPE (or ConstantPE for empty)."""
        if self._mix_pe is not None:
            return self._mix_pe.channel_count()
        return None
    
    def _compute_extent(self) -> Extent:
        """
        Return extent determined by start of first input and end of last input.
        
        Extent = (start of first input, end of last input)
        """
        if not self._sequence:
            # Empty sequence: infinite extent (ConstantPE)
            return Extent(None, None)
        
        # Get start time of first item
        first_item = self._sequence[0]
        first_pe = first_item[0]
        first_start = first_item[1]
        
        # Get the extent of the first PE (in its local time)
        first_pe_extent = first_pe.extent()
        
        # Calculate start time in global time
        # If first PE has infinite backward extent, SequencePE also has infinite backward extent
        if first_pe_extent.start is None:
            # First PE extends infinitely backward - SequencePE also extends backward infinitely
            sequence_start = None
        else:
            # First PE has finite start - SequencePE starts when first item starts
            sequence_start = first_start
        
        # Get end time of last item
        last_item = self._sequence[-1]
        last_pe = last_item[0]
        last_start = last_item[1]
        
        # Get the extent of the last PE (in its local time)
        last_pe_extent = last_pe.extent()
        
        # Calculate end time in global time
        if last_pe_extent.end is not None:
            # Last PE has finite extent
            last_end = last_start + last_pe_extent.end
        else:
            # Last PE has infinite extent
            last_end = None
        
        return Extent(sequence_start, last_end)
    
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
        return f"SequencePE({count} items)"
