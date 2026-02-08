"""
PortamentoPE - creates portamento (pitch gliding) effects between notes.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from typing import Optional

from pygmu2.processing_element import SourcePE, ProcessingElement
from pygmu2.extent import Extent, ExtendMode
from pygmu2.snippet import Snippet
from pygmu2.sequence_pe import SequencePE
from pygmu2.piecewise_pe import PiecewisePE
from pygmu2.constant_pe import ConstantPE
from pygmu2.crop_pe import CropPE
from pygmu2.logger import get_logger

logger = get_logger(__name__)


class PortamentoPE(SourcePE):
    """
    A SourcePE that creates portamento (pitch gliding) effects between notes.
    
    Takes a list of (pitch, sample_index, duration) tuples and emits a stream
    of pitch values. When the pitch changes between notes, it ramps smoothly
    from the previous pitch to the next over an adaptive ramp time.
    
    The first pitch in the list determines the initial pitch - no portamento
    is required on the first note.
    
    Behavior with disjoint or overlapping notes:
    
    Disjoint notes (gap between notes):
        When notes have a gap between them, the pitch holds the previous note's
        value during the gap, then transitions when the next note starts.
        
        Example: [(69, 0, 500), (73, 1000, 500)]
        - Note 0: pitch=69 from t=0 to t=500 (ends at 500)
        - Gap: t=500 to t=1000 (500 samples)
        - Note 1: pitch=73 from t=1000 to t=1500
        - Behavior:
          * t=0 to t=1000: pitch=69 (held from first note, before ramp starts)
          * t=1000: ramp begins, transitions from 69 to 73
          * After ramp: pitch=73 (held for note 1's duration)
    
    Overlapping notes:
        When notes overlap, the pitch transitions during the overlap period.
        
        Example: [(69, 0, 1500), (73, 1000, 1500)]
        - Note 0: pitch=69 from t=0 to t=1500
        - Note 1: pitch=73 from t=1000 to t=2500
        - Overlap: t=1000 to t=1500 (500 samples)
        - Behavior:
          * t=0 to t=1000: pitch=69 (held from first note, before ramp starts)
          * t=1000: ramp begins, transitions from 69 to 73 during overlap
          * After ramp completes: pitch=73 (held for note 1's duration)
    
    Behavior outside note range:
        PortamentoPE has infinite extent and holds pitch values outside the note range:
        
        Before first note:
            The pitch holds the first note's pitch value for all times before the first
            note starts. This is because the first ramp (transitioning to the second note)
            uses ExtendMode.HOLD_BOTH and holds the start value (first note's pitch) before
            the ramp begins.
            
            Example: For notes [(69, 0, 500), (73, 1000, 500)]
            - At time -500: pitch=69.0 (first note's pitch, held)
            - At time -100: pitch=69.0 (first note's pitch, held)
        
        After last note:
            The pitch holds the last note's pitch value for all times after the last note
            ends. This is because the last ramp has infinite extent (not cropped) and uses
            ExtendMode.HOLD_BOTH, holding the end value (last note's pitch) after the ramp
            completes.
            
            Example: For notes [(69, 0, 500), (73, 1000, 500)]
            - Last note ends at: 1000 + 500 = 1500
            - At time 2000: pitch=73.0 (last note's pitch, held)
            - At time 5000: pitch=73.0 (last note's pitch, held)
    
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
        # Using seconds (resolved at construction time)
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
        
        # Sequence is built at construction time (sample_rate is globally available).
        self._sequence_pe: Optional[SequencePE] = None
        self._build_sequence()
    
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
    
    def _build_sequence(self) -> None:
        """Build the internal SequencePE based on note list and timing params."""
        # Zero notes: should have been caught in __init__, but handle gracefully
        if not self._notes:
            raise ValueError("PortamentoPE: notes list cannot be empty")
        
        # One note: return ConstantPE delayed to note's start (infinite extent, consistent with N>=2)
        if len(self._notes) == 1:
            pitch, start, _duration = self._notes[0]
            from pygmu2.delay_pe import DelayPE
            constant_pe = ConstantPE(pitch, channels=self._channels)
            # Delay to note's start time (infinite extent, like last ramp in N>=2 case)
            delayed = DelayPE(constant_pe, delay=start)
            self._sequence_pe = delayed
            return
        
        # N notes (N >= 2): create N-1 ramps with HOLD_BOTH
        # Resolve max_ramp_samples using _time_to_samples pattern
        max_ramp_samples_resolved = self._time_to_samples(
            samples=self._max_ramp_samples,
            seconds=self._max_ramp_seconds,
            name="max_ramp",
        )
        # Ensure at least 1 sample
        max_ramp_samples_resolved = max(1, max_ramp_samples_resolved)
        
        sequence_items = []
        
        # Create N-1 ramps for transitions between notes
        # Each ramp transitions from previous note's pitch to current note's pitch
        # Ramps start at the current note's start time (when transition begins)
        # HOLD_BOTH ensures:
        #   - First ramp holds note 0's pitch before it starts (covering note 0's period)
        #   - Each ramp holds its end pitch after completion (covering the note's duration)
        #   - Last ramp holds last note's pitch indefinitely (infinite extent)
        for i in range(len(self._notes) - 1):
            prev_pitch, prev_start, prev_duration = self._notes[i]
            curr_pitch, curr_start, curr_duration = self._notes[i + 1]
            
            # Calculate when next ramp starts (if there is one)
            next_ramp_start = None
            if i < len(self._notes) - 2:
                next_ramp_start = self._notes[i + 2][1]  # Start time of note after next
            
            # Debug: log note transition
            from pygmu2.conversions import pitch_to_freq
            prev_freq = pitch_to_freq(prev_pitch)
            curr_freq = pitch_to_freq(curr_pitch)
            logger.debug(
                f"PortamentoPE: Note transition {i}->{i+1} - "
                f"prev_pitch={prev_pitch:.2f} (freq={prev_freq:.2f} Hz), "
                f"curr_pitch={curr_pitch:.2f} (freq={curr_freq:.2f} Hz), "
                f"prev_start={prev_start}, curr_start={curr_start}, curr_duration={curr_duration} samples"
            )
            
            # Adaptive ramp duration: min(max_ramp_time, current_note_duration * ramp_fraction)
            # The ramp duration is limited by the current note's duration
            adaptive_ramp_samples = min(max_ramp_samples_resolved, int(round(curr_duration * self._ramp_fraction)))
            # Ensure at least 1 sample
            ramp_duration = max(1, adaptive_ramp_samples)
            
            logger.debug(
                f"PortamentoPE: Creating ramp - duration={ramp_duration} samples, "
                f"start_value={prev_pitch:.2f}, end_value={curr_pitch:.2f}, channels={self._channels}"
            )
            
            # Create ramp from previous pitch to current pitch
            # With ExtendMode.HOLD_BOTH:
            #   - Before ramp start: holds prev_pitch (covers previous note's duration)
            #   - During ramp: ramps from prev_pitch to curr_pitch
            #   - After ramp: holds curr_pitch (covers current note's duration)
            ramp = PiecewisePE(
                [(0, prev_pitch), (ramp_duration, curr_pitch)],
                extend_mode=ExtendMode.HOLD_BOTH,
                channels=self._channels,
            )
            
            # Crop ramps to prevent unwanted contributions:
            # - First ramp: crop end to start of next note (if there is one), no start crop (holds first note's pitch)
            # - Middle ramps: crop start (to time 0) and end (to next note's start)
            # - Last ramp: crop start (to time 0), no end crop (infinite extent)
            # Exception: single ramp (2 notes total): no cropping (infinite extent on both sides)
            if len(self._notes) == 2:
                # Single ramp: no cropping - infinite extent on both sides
                # Holds first note's pitch before ramp, last note's pitch after ramp
                cropped_ramp = ramp
            elif i == 0:
                # First ramp (3+ notes): crop end to next ramp's start
                # No start crop - needs to hold first note's pitch before ramp starts
                # Use HOLD_FIRST to hold the ramp's start value (first note's pitch) before crop window
                crop_duration = next_ramp_start - curr_start
                cropped_ramp = CropPE(ramp, 0, crop_duration, extend_mode=ExtendMode.HOLD_FIRST)
            elif i == len(self._notes) - 2:
                # Last ramp: crop start (to time 0), no end crop (infinite extent)
                cropped_ramp = CropPE(ramp, 0, None, extend_mode=ExtendMode.ZERO)
            else:
                # Middle ramps: crop start (to time 0) and end (to next note's start)
                crop_duration = next_ramp_start - curr_start
                cropped_ramp = CropPE(ramp, 0, (crop_duration) - (0), extend_mode=ExtendMode.ZERO)
            
            # Ramp starts at current note's start time
            sequence_items.append((cropped_ramp, curr_start))
        
        # Create SequencePE (always delays and mixes all inputs)
        # SequencePE delays each ramp to its start time and mixes them together
        # The last ramp is never cropped, so it has infinite extent (HOLD_BOTH holds last pitch)
        # This gives PortamentoPE infinite extent with HOLD_BOTH behavior
        self._sequence_pe = SequencePE(sequence_items, channels=self._channels)
    
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
            snippet = self._sequence_pe.render(start, duration)
            # Debug: log sample values at key points
            if duration > 0:
                import numpy as np
                # Log first, middle, and last sample
                first_pitch = snippet.data[0, 0] if snippet.data.shape[0] > 0 else 0
                mid_idx = duration // 2
                mid_pitch = snippet.data[mid_idx, 0] if mid_idx < snippet.data.shape[0] else 0
                last_pitch = snippet.data[-1, 0] if snippet.data.shape[0] > 0 else 0
                
                from pygmu2.conversions import pitch_to_freq
                logger.debug(
                    f"PortamentoPE._render: start={start}, duration={duration}, "
                    f"first_pitch={first_pitch:.2f} (freq={pitch_to_freq(first_pitch):.2f} Hz), "
                    f"mid_pitch={mid_pitch:.2f} (freq={pitch_to_freq(mid_pitch):.2f} Hz), "
                    f"last_pitch={last_pitch:.2f} (freq={pitch_to_freq(last_pitch):.2f} Hz), "
                    f"data_shape={snippet.data.shape}"
                )
            return snippet
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
