"""
PortamentoPE - creates portamento (pitch gliding) effects between notes.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

from typing import Optional

from pygmu2.processing_element import SourcePE, ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.sequence_pe import SequencePE
from pygmu2.ramp_pe import RampPE
from pygmu2.constant_pe import ConstantPE


class PortamentoPE(SourcePE):
    """
    A SourcePE that creates portamento (pitch gliding) effects between notes.
    
    Takes a list of (pitch, sample_index, duration) tuples and emits a stream
    of pitch values. When the pitch changes between notes, it ramps smoothly
    from the previous pitch to the next over an adaptive ramp time.
    
    The first pitch in the list determines the initial pitch - no portamento
    is required on the first note.
    
    Args:
        notes: List of (pitch, sample_index, duration) tuples where:
               - pitch: The target pitch value, typically MIDI note numbers (float)
                        e.g., 69.0 for A4, 73.0 for C#5, 76.0 for E5
               - sample_index: When this note starts (in samples)
               - duration: Duration of this note (in samples)
        max_ramp_samples: Maximum ramp time in samples (optional)
        max_ramp_seconds: Maximum ramp time in seconds (optional, default: 0.1)
                          Specify either max_ramp_samples or max_ramp_seconds, not both.
        ramp_fraction: Fraction of note duration to use for ramp if note is short
                      (default: 0.3, meaning ramp uses up to 30% of note duration)
        channels: Number of output channels (default: 1)
    
    Example:
        # Portamento between three MIDI notes (A4, C#5, E5)
        notes = [
            (69.0, 0, 1000),      # A4 (MIDI 69) at t=0, duration 1000 samples
            (73.0, 1000, 1000),   # C#5 (MIDI 73) at t=1000, duration 1000 samples
            (76.0, 2000, 1000),   # E5 (MIDI 76) at t=2000, duration 1000 samples
        ]
        # Using seconds (resolved at configure time)
        pitch_stream = PortamentoPE(notes, max_ramp_seconds=0.05)
        
        # Or using samples directly
        pitch_stream = PortamentoPE(notes, max_ramp_samples=2205)  # ~0.05s at 44.1kHz
        
        # Convert MIDI pitches to frequencies for use with oscillators
        from pygmu2 import pitch_to_freq
        freq_stream = TransformPE(pitch_stream, pitch_to_freq)
        synth_stream = SinePE(frequency=freq_stream)
    """
    
    def __init__(
        self,
        notes: list[tuple[float, int, int]],
        max_ramp_samples: Optional[int] = None,
        max_ramp_seconds: Optional[float] = None,
        ramp_fraction: float = 0.3,
        channels: int = 1,
    ):
        if not notes:
            raise ValueError("PortamentoPE: notes list cannot be empty")
        
        # Resolve max_ramp: exactly one of samples or seconds must be provided
        if max_ramp_samples is None and max_ramp_seconds is None:
            # Default to seconds if neither provided
            max_ramp_seconds = 0.1
        
        if max_ramp_samples is not None and max_ramp_seconds is not None:
            raise ValueError(
                "PortamentoPE: specify either max_ramp_samples or max_ramp_seconds, not both "
                f"(got max_ramp_samples={max_ramp_samples}, max_ramp_seconds={max_ramp_seconds})"
            )
        
        if max_ramp_samples is not None and max_ramp_samples < 0:
            raise ValueError(f"PortamentoPE: max_ramp_samples must be non-negative (got {max_ramp_samples})")
        
        if max_ramp_seconds is not None and max_ramp_seconds < 0:
            raise ValueError(f"PortamentoPE: max_ramp_seconds must be non-negative (got {max_ramp_seconds})")
        
        if not (0.0 <= ramp_fraction <= 1.0):
            raise ValueError(f"PortamentoPE: ramp_fraction must be between 0 and 1 (got {ramp_fraction})")
        
        if channels < 1:
            raise ValueError(f"PortamentoPE: channels must be >= 1 (got {channels})")
        
        self._notes = sorted(notes, key=lambda x: x[1])  # Sort by sample_index
        self._max_ramp_samples = int(max_ramp_samples) if max_ramp_samples is not None else None
        self._max_ramp_seconds = float(max_ramp_seconds) if max_ramp_seconds is not None else None
        self._ramp_fraction = float(ramp_fraction)
        self._channels = int(channels)
        
        # Sequence will be built in configure() when sample_rate is available
        self._sequence_pe: Optional[SequencePE] = None
    
    @property
    def notes(self) -> list[tuple[float, int, int]]:
        """The list of (pitch, sample_index, duration) tuples (sorted by sample_index)."""
        return self._notes.copy()
    
    @property
    def max_ramp_samples(self) -> Optional[int]:
        """Maximum ramp time in samples (if specified)."""
        return self._max_ramp_samples
    
    @property
    def max_ramp_seconds(self) -> Optional[float]:
        """Maximum ramp time in seconds (if specified)."""
        return self._max_ramp_seconds
    
    @property
    def ramp_fraction(self) -> float:
        """Fraction of note duration used for ramp on short notes."""
        return self._ramp_fraction
    
    def configure(self, sample_rate: int) -> None:
        """Configure the portamento PE with sample rate and build the sequence."""
        super().configure(sample_rate)
        
        # Resolve max_ramp_samples using _time_to_samples pattern
        max_ramp_samples_resolved = self._time_to_samples(
            samples=self._max_ramp_samples,
            seconds=self._max_ramp_seconds,
            name="max_ramp",
        )
        # Ensure at least 1 sample
        max_ramp_samples_resolved = max(1, max_ramp_samples_resolved)
        
        # Build sequence of PEs now that we have sample_rate
        sequence_items = []
        
        if not self._notes:
            # Should not happen (checked in __init__), but handle gracefully
            self._sequence_pe = ConstantPE(0.0, channels=self._channels)
            self._sequence_pe.configure(sample_rate)
            return
        
        # First note: output constant pitch from time 0 (before first note starts)
        # The first pitch determines the initial pitch - no portamento on first note
        first_pitch, first_start, _first_duration = self._notes[0]
        sequence_items.append((ConstantPE(first_pitch, channels=self._channels), 0))
        
        # Process transitions between notes
        for i in range(len(self._notes) - 1):
            prev_pitch, prev_start, prev_duration = self._notes[i]
            curr_pitch, curr_start, curr_duration = self._notes[i + 1]
            
            if abs(curr_pitch - prev_pitch) < 1e-6:
                # Same pitch: no portamento needed, use constant starting at current note
                sequence_items.append((ConstantPE(curr_pitch, channels=self._channels), curr_start))
            else:
                # Different pitch: create ramp with adaptive duration
                # Adaptive ramp duration: min(max_ramp_time, note_duration * ramp_fraction)
                adaptive_ramp_samples = min(max_ramp_samples_resolved, int(round(curr_duration * self._ramp_fraction)))
                # Ensure at least 1 sample
                ramp_duration = max(1, adaptive_ramp_samples)
                
                # Create ramp from previous pitch to current pitch
                # Ramp starts at current note's start_time
                # With hold_extents=True:
                #   - Before ramp start: holds prev_pitch (from hold_extents)
                #   - During ramp: ramps from prev_pitch to curr_pitch
                #   - After ramp: holds curr_pitch (from hold_extents)
                ramp = RampPE(
                    start_value=prev_pitch,
                    end_value=curr_pitch,
                    duration=ramp_duration,
                    channels=self._channels,
                    hold_extents=True,
                )
                sequence_items.append((ramp, curr_start))
        
        # Use SequencePE with overlap=False
        # With overlap=False, SequencePE will crop each PE to prevent overlap with the next.
        # This ensures that:
        # - The first ConstantPE outputs first_pitch from 0 until the next item starts
        # - Each subsequent item (ramp or constant) takes over at its start_time
        # - RampPE with hold_extents=True will hold values before/after the ramp, ensuring
        #   smooth transitions without gaps
        self._sequence_pe = SequencePE(sequence_items, channels=self._channels, overlap=False)
        self._sequence_pe.configure(sample_rate)
    
    def inputs(self) -> list[ProcessingElement]:
        """Return inputs (the internal SequencePE)."""
        return [self._sequence_pe] if self._sequence_pe is not None else []
    
    def is_pure(self) -> bool:
        """PortamentoPE is pure - it's a stateless composition."""
        return True
    
    def channel_count(self) -> Optional[int]:
        """Return channel count."""
        return self._channels
    
    def _compute_extent(self) -> Extent:
        """Return extent from internal SequencePE."""
        if self._sequence_pe is not None:
            return self._sequence_pe.extent()
        return Extent(None, None)
    
    def _render(self, start: int, duration: int) -> Snippet:
        """Render portamento pitch values."""
        if self._sequence_pe is not None:
            return self._sequence_pe.render(start, duration)
        return Snippet.from_zeros(start, duration, self._channels)
    
    def __repr__(self) -> str:
        count = len(self._notes)
        if self._max_ramp_samples is not None:
            ramp_str = f"max_ramp_samples={self._max_ramp_samples}"
        else:
            ramp_str = f"max_ramp_seconds={self._max_ramp_seconds}"
        return (
            f"PortamentoPE({count} notes, {ramp_str}, "
            f"ramp_fraction={self._ramp_fraction}, channels={self._channels})"
        )
