"""
AdsrGatedPE, AdsrTriggeredPE - ADSR envelope generators.

Uses segment-based vectorized rendering for efficiency.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.snippet import Snippet
from pygmu2.extent import Extent
from pygmu2.config import get_sample_rate
from pygmu2.gate_signal import GateSignal
from pygmu2.trigger_signal import TriggerSignal

from pygmu2.logger import get_logger

logger = get_logger(__name__)
logger.setLevel("WARN")

# Envelope states. These are string constants to make logs readable.
IDLE = "idle"
ATTACK = "attack"
DECAY = "decay"
SUSTAIN = "sustain"
RELEASE = "release"


def _generate_ramp(
    output: np.ndarray,
    starting_value: float,
    slope: float,
    offset: int,
    length: int,
) -> tuple[int, float]:
    """
    Write a linear ramp into output[offset:offset+length] and return
    (offset + length, starting_value + slope * length).

    output[offset + i] = starting_value + i * slope   for i in 0..length-1
    """
    output[offset : offset + length] = (
        starting_value + np.arange(length, dtype=np.float64) * slope
    ).astype(np.float32)
    return (offset + length, starting_value + slope * length)


class AdsrGatedPE(ProcessingElement):
    """
    Gate-driven ADSR envelope generator.

    This ProcessingElement outputs a *mono control signal* (shape in [0..1]) that
    can be used to modulate gain (a VCA), filter cutoff, etc.

    Semantics:
      - Input is a GateSignal (values 0 or 1).
      - Rising edge (0 -> 1) starts a new envelope cycle (Attack from current state).
      - While the gate stays high: Attack -> Decay -> Sustain, holding sustain level.
      - Falling edge (1 -> 0) starts Release from the current envelope value.
      - When Release reaches 0, returns to IDLE.

    Notes:
      - This PE is stateful and therefore must be rendered
        with contiguous render requests.
      - Internally it uses segment-based vectorized rendering: the buffer is split
        at gate transitions and ADSR phase boundaries, then filled with numpy slices.

    Args:
        gate: GateSignal controlling the envelope (0 or 1).
        attack_time: seconds to ramp from 0 to 1.
        decay_time: seconds to ramp from 1 down to sustain_level.
        sustain_level: steady-state level in [0..1] while gate is high.
        release_time: seconds to ramp from sustain_level down to 0.
    """

    def __init__(
        self,
        gate: GateSignal,
        attack_time: float = 0.1,
        decay_time: float = 0.1,
        sustain_level: float = 0.5,
        release_time: float = 0.1,
    ):
        self._gate = gate
        self._attack_time = float(attack_time)
        self._decay_time = float(decay_time)
        self._sustain_level = float(sustain_level)
        self._release_time = float(release_time)

        # Precompute slopes (dv/dt per sample).
        # These are "delta per sample" increments applied to self._env.
        #
        # Attack: 0.0 -> 1.0
        # Decay:  1.0 -> sustain_level (dvdt is negative unless sustain_level==1)
        # Release: sustain_level -> 0.0 (dvdt is negative unless sustain_level==0)
        sr = float(get_sample_rate())
        self._attack_dvdt = (1.0 - 0.0) / (self._attack_time * sr)
        self._decay_dvdt = (self._sustain_level - 1.0) / (self._decay_time * sr)
        self._release_dvdt = (0.0 - self._sustain_level) / (self._release_time * sr)

        # Initialize runtime state.
        self._reset_state()

    stateful = True

    def channel_count(self) -> int:
        # Mono control signal
        return 1

    def inputs(self) -> list[ProcessingElement]:
        # This PE depends on the gate input.
        return [self._gate]

    def _compute_extent(self) -> Extent:
        # Envelope is defined wherever the gate is defined.
        return self._gate.extent()

    def _on_start(self) -> None:
        # Called by renderer when playback begins.
        self._reset_state()

    def _on_stop(self) -> None:
        # Called by renderer when playback ends.
        self._reset_state()

    def _reset_state(self):
        # Runtime state:
        #   - _state: one of IDLE/ATTACK/DECAY/SUSTAIN/RELEASE
        #   - _env: current envelope level (float)
        #   - _prev_gate: last gate sample, used to detect edges
        self._state = IDLE
        self._env = 0.0
        self._prev_gate = 0

    def _update_state(self, now, new_state):
        # Debug helper to log state transitions with absolute sample index.
        logger.debug(f"{now}: {self._state} => {new_state}")
        self._state = new_state

    def _render(self, start: int, duration: int) -> Snippet:
        """
        Render `duration` samples starting at absolute sample index `start`.

        Output:
            Snippet(start, out) where out is a 1-D float32 array of length duration.
        """
        # Place to write the result.
        out = np.zeros(duration, dtype=np.float32)
        # GateSignal data is 0 or 1; flatten to 1-D in case it comes back as (N,1).
        raw = self._gate.render(start=start, duration=duration).data
        gate_data = raw[:, 0] if raw.ndim > 1 else raw

        # Repeatedly call _render_segment until we've rendered to the end of
        # this render buffer.
        cursor = 0
        while cursor < duration:
            cursor = self._render_segment(start, cursor, duration, out, gate_data)

        return Snippet(start, out)

    def _render_segment(
        self,
        start: int,
        cursor: int,
        duration: int,
        out_data: np.ndarray,
        gate_data: np.ndarray,
    ) -> int:
        """
        Process one contiguous region — either a constant-gate segment or the
        portion of the current ADSR phase that fits within it — and return the
        updated cursor position.
        """
        # Detect gate edge at this cursor and update state immediately.
        curr_gate = gate_data[cursor]
        if curr_gate != self._prev_gate:
            self._state = ATTACK if curr_gate > self._prev_gate else RELEASE
        self._prev_gate = curr_gate

        # End of the current constant-gate region.
        gate_end = self._next_gate_event(cursor, duration, gate_data)
        seg_len = gate_end - cursor

        # Constant states: fill the whole gate segment and return.
        if self._state == IDLE:
            out_data[cursor:gate_end] = 0.0
            self._env = 0.0
            return gate_end

        if self._state == SUSTAIN:
            out_data[cursor:gate_end] = self._sustain_level
            self._env = self._sustain_level
            return gate_end

        # Ramping states: select slope, threshold, and successor state.
        if self._state == ATTACK:
            dvdt, threshold, next_state = self._attack_dvdt, 1.0, DECAY
        elif self._state == DECAY:
            dvdt, threshold, next_state = self._decay_dvdt, self._sustain_level, SUSTAIN
        else:  # RELEASE
            dvdt, threshold, next_state = self._release_dvdt, 0.0, IDLE

        # Number of samples until the state transition.
        # Formula: emit env₀ + k×dvdt at step k; crossing at post-step k+1 ≥ threshold
        # → n_in_state = ceil(T) where T = (threshold - env₀) / dvdt.
        if dvdt == 0.0:
            n_in_state = 1  # sustain_level == threshold → instant transition
        else:
            T = (threshold - self._env) / dvdt
            n_in_state = max(1, int(np.ceil(T)))

        n = min(seg_len, n_in_state)

        # Vectorised ramp fill; advances cursor and env.
        cursor, self._env = _generate_ramp(out_data, self._env, dvdt, cursor, n)

        # Clamp and transition if the phase completed within this segment.
        if n == n_in_state:
            self._env = threshold
            self._state = next_state

        return cursor

    def _next_gate_event(
        self, cursor: int, duration: int, gate_data: np.ndarray
    ) -> int:
        """
        Return the index of the first gate transition strictly after `cursor`,
        or `duration` if the gate value is constant for the rest of the buffer.
        """
        current = gate_data[cursor]
        changes = np.where(gate_data[cursor + 1 :] != current)[0]
        return cursor + 1 + int(changes[0]) if len(changes) > 0 else duration


class AdsrTriggeredPE(ProcessingElement):
    """
    Trigger-driven one-shot ADSR envelope generator.

    This ProcessingElement outputs a *mono control signal* in [0..1].

    Semantics:
      - Input is a TriggerSignal (typically impulses, positive values indicate events).
      - When trigger > 0 at a sample: restart the ADSR cycle (Attack begins immediately).
      - Progresses Attack -> Decay -> Sustain for a fixed sustain_time -> Release -> Idle.
      - A new trigger during any phase restarts the cycle.

    Args:
        trigger: TriggerSignal; any positive sample triggers a restart.
        attack_time: seconds to ramp from 0 to 1.
        decay_time: seconds to ramp from 1 down to sustain_level.
        sustain_time: seconds to hold sustain_level (fixed duration, unlike gated ADSR).
        sustain_level: steady-state level in [0..1] during sustain phase.
        release_time: seconds to ramp down to 0.
    """

    def __init__(
        self,
        trigger: TriggerSignal,
        attack_time: float = 0.1,
        decay_time: float = 0.1,
        sustain_time: float = 0.5,
        sustain_level: float = 0.5,
        release_time: float = 0.1,
    ):
        self._trigger = trigger
        self._attack_time = float(attack_time)
        self._decay_time = float(decay_time)
        self._sustain_time = float(sustain_time)
        self._sustain_level = float(sustain_level)
        self._release_time = float(release_time)

        # Precompute slopes (dv/dt per sample).
        sr = float(get_sample_rate())
        self._attack_dvdt = (1.0 - 0.0) / (self._attack_time * sr)
        self._decay_dvdt = (self._sustain_level - 1.0) / (self._decay_time * sr)
        self._release_dvdt = (0.0 - self._sustain_level) / (self._release_time * sr)

        # Convert sustain_time (seconds) into an absolute sample count.
        # We count down by comparing absolute 'now' to an end time.
        self._sustain_samples = int(round(self._sustain_time * sr))

        self._reset_state()

    stateful = True

    def channel_count(self) -> int:
        return 1

    def inputs(self) -> list[ProcessingElement]:
        return [self._trigger]

    def _compute_extent(self) -> Extent:
        return self._trigger.extent()

    def _on_start(self) -> None:
        self._reset_state()

    def _on_stop(self) -> None:
        self._reset_state()

    def _reset_state(self):
        # Runtime state:
        #   - _state: current ADSR phase
        #   - _env: current envelope level
        #   - _sustain_ends_at: absolute sample index when sustain should end
        self._state = IDLE
        self._env = 0.0
        self._sustain_ends_at = 0

    def _update_state(self, now, new_state):
        logger.debug(f"{now}: {self._state} => {new_state}")
        self._state = new_state

    def _render(self, start: int, duration: int) -> Snippet:
        """
        Render `duration` samples starting at absolute sample index `start`.

        Output:
            Snippet(start, out) where out is a 1-D float32 array of length duration.
        """
        out = np.zeros(duration, dtype=np.float32)
        # Flatten to 1-D in case trigger data comes back as (N, 1).
        raw = self._trigger.render(start=start, duration=duration).data
        trigger_data = raw[:, 0] if raw.ndim > 1 else raw

        cursor = 0
        while cursor < duration:
            cursor = self._render_segment(start, cursor, duration, out, trigger_data)

        return Snippet(start, out)

    def _next_trigger_event(
        self, cursor: int, duration: int, trigger_data: np.ndarray
    ) -> int:
        """
        Return the index of the first positive trigger sample STRICTLY AFTER
        `cursor`, or `duration` if no such sample exists.

        We skip `cursor` itself because the caller already handles any trigger
        present at that position before calling this method.
        """
        if cursor + 1 >= duration:
            return duration
        hits = np.where(trigger_data[cursor + 1 :] > 0.0)[0]
        return cursor + 1 + int(hits[0]) if len(hits) > 0 else duration

    def _render_segment(
        self,
        start: int,
        cursor: int,
        duration: int,
        out_data: np.ndarray,
        trigger_data: np.ndarray,
    ) -> int:
        """
        Process one contiguous region — either a constant-trigger segment or the
        portion of the current ADSR phase that fits within it — and return the
        updated cursor position.
        """
        # Detect trigger at cursor → restart to ATTACK from current env level.
        if trigger_data[cursor] > 0.0:
            self._state = ATTACK

        # Find the next trigger event (after cursor) to bound this segment.
        trig_end = self._next_trigger_event(cursor, duration, trigger_data)

        # IDLE: fill with silence up to the next trigger.
        if self._state == IDLE:
            out_data[cursor:trig_end] = 0.0
            self._env = 0.0
            return trig_end

        # SUSTAIN: fill at sustain_level, bounded by next trigger OR the timer.
        if self._state == SUSTAIN:
            sustain_end = self._sustain_ends_at - start  # convert to buffer-relative
            seg_end = min(trig_end, sustain_end)
            seg_end = max(cursor + 1, seg_end)  # always advance at least one sample
            out_data[cursor:seg_end] = self._sustain_level
            self._env = self._sustain_level
            if seg_end >= sustain_end:
                self._state = RELEASE
            return seg_end

        # Ramping states: ATTACK, DECAY, RELEASE.
        seg_len = trig_end - cursor

        if self._state == ATTACK:
            dvdt, threshold, next_state = self._attack_dvdt, 1.0, DECAY
        elif self._state == DECAY:
            dvdt, threshold, next_state = self._decay_dvdt, self._sustain_level, SUSTAIN
        else:  # RELEASE
            dvdt, threshold, next_state = self._release_dvdt, 0.0, IDLE

        # Number of samples until the state transition.
        if dvdt == 0.0:
            n_in_state = 1
        else:
            T = (threshold - self._env) / dvdt
            n_in_state = max(1, int(np.ceil(T)))

        n = min(seg_len, n_in_state)

        # Vectorised ramp fill; advances cursor and env.
        cursor, self._env = _generate_ramp(out_data, self._env, dvdt, cursor, n)

        # Clamp and transition if the phase completed within this segment.
        if n == n_in_state:
            self._env = threshold
            if next_state == SUSTAIN:
                # Start the sustain countdown from the first SUSTAIN sample (cursor).
                self._sustain_ends_at = start + cursor + self._sustain_samples
            self._state = next_state

        return cursor
