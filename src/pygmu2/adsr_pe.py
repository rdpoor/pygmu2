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
      - This PE is stateful (is_pure() == False) and therefore must be rendered
        with contiguous render requests.
      - Internally it is implemented as a 1-sample-per-iteration state machine.
        (Vectorization is possible but not shown here.)

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
        release_time: float = 0.1
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

    def is_pure(self) -> bool:
        # Envelope evolution depends on internal state, not solely on (start, duration).
        return False

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
        logger.debug(f'{now}: {self._state} => {new_state}')
        self._state = new_state

    def _render(self, start: int, duration: int) -> Snippet:
        """
        Render `duration` samples starting at absolute sample index `start`.

        Output:
            Snippet(start, out) where out is a 1-D float32 array of length duration.
        """
        out = np.zeros(duration, dtype=np.float32)

        # GateSignal should produce shape (duration, 1) in pygmu2.
        # This code assumes gate_data[cursor] returns scalar 0 or 1.
        gate_data = self._gate.render(start=start, duration=duration).data

        for cursor in range(duration):
            # Emit the envelope level for this sample.
            # (Important: this outputs the *current* env, then advances the state.)
            out[cursor] = self._env
            now = start + cursor

            # Detect gate transitions by comparing current gate value to previous.
            # Rising edge triggers ATTACK; falling edge triggers RELEASE.
            curr_gate = gate_data[cursor]
            is_new_attack = (self._prev_gate == 0 and curr_gate == 1)
            is_new_release = (self._prev_gate == 1 and curr_gate == 0)
            self._prev_gate = curr_gate

            # Apply event-driven state changes immediately.
            if is_new_attack:
                self._update_state(now, ATTACK)
            elif is_new_release:
                self._update_state(now, RELEASE)

            # State machine: update self._env and possibly transition.
            if self._state == IDLE:
                # Gate is low (or envelope finished). Force env to 0.
                self._env = 0.0

            elif self._state == ATTACK:
                # Linear ramp up to 1.0 using precomputed slope.
                self._env += self._attack_dvdt
                if self._env >= 1.0:
                    # Clamp and transition to decay.
                    self._env = 1.0
                    self._update_state(now, DECAY)

            elif self._state == DECAY:
                # Linear ramp down to sustain level.
                self._env += self._decay_dvdt
                if self._env <= self._sustain_level:
                    # Clamp and transition to sustain.
                    self._env = self._sustain_level
                    self._update_state(now, SUSTAIN)

            elif self._state == SUSTAIN:
                # Hold sustain level while gate remains high.
                # (In this gated variant, sustain ends only on a gate falling edge.)
                self._env = self._sustain_level

            elif self._state == RELEASE:
                # Linear ramp down to 0.0.
                # Note: dvdt was computed assuming a release starting at sustain_level,
                # but we may enter release from any current envelope value. Using the same
                # slope gives a constant-time release only when releasing from sustain.
                # If you want constant-time release from arbitrary env, you would compute
                # dvdt based on current env at release start.
                self._env += self._release_dvdt
                if self._env <= 0.0:
                    self._env = 0.0
                    self._update_state(now, IDLE)

        # Snippet expects (start, data). In your earlier codebase, Snippet(start, out)
        # is valid for 1-D mono, but many PEs use (N,1). Adjust to your convention as needed.
        return Snippet(start, out)


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
        release_time: float = 0.1
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

        self.reset_state()

    def is_pure(self) -> bool:
        return False

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
        logger.debug(f'{now}: {self._state} => {new_state}')
        self._state = new_state

    def _render(self, start: int, duration: int) -> Snippet:
        """
        Render `duration` envelope samples.

        Important: If trigger pulses occur at block boundaries, this per-sample loop
        guarantees that the envelope restart happens on the exact trigger sample.
        """
        out = np.zeros(duration, dtype=np.float32)

        # TriggerSignal should produce shape (duration, 1). We treat trigger>0 as event.
        trigger_data = self._trigger.render(start=start, duration=duration).data

        for cursor in range(duration):
            out[cursor] = self._env
            now = start + cursor

            # Trigger restart: any positive sample restarts the ADSR cycle.
            # This is intentionally simple; if you later encode +/- edges you can
            # distinguish rising vs falling events.
            is_new_attack = (trigger_data[cursor] > 0.0)

            if is_new_attack:
                # Restart the envelope immediately on the trigger sample.
                # This implementation does not force env=0; it begins attack from the
                # current env. If you want "restart from 0", set self._env = 0.0 here.
                self._update_state(now, ATTACK)

            if self._state == IDLE:
                self._env = 0.0

            elif self._state == ATTACK:
                self._env += self._attack_dvdt
                if self._env >= 1.0:
                    self._env = 1.0
                    self._update_state(now, DECAY)

            elif self._state == DECAY:
                self._env += self._decay_dvdt
                if self._env <= self._sustain_level:
                    self._env = self._sustain_level
                    # Schedule sustain end time in absolute samples.
                    self._sustain_ends_at = now + self._sustain_samples
                    self._update_state(now, SUSTAIN)

            elif self._state == SUSTAIN:
                # Hold sustain level until the sustain timer expires.
                self._env = self._sustain_level
                if now >= self._sustain_ends_at:
                    self._update_state(now, RELEASE)

            elif self._state == RELEASE:
                self._env += self._release_dvdt
                if self._env <= 0.0:
                    self._env = 0.0
                    self._update_state(now, IDLE)

        return Snippet(start, out)
