# src/pygmu2/adsr_signal.py
#
# ADSR envelope generators for the GateSignal / TriggerSignal system.
#
# Public classes:
#   - AdsrGateSignal: gate-driven ADSR (attack/decay/sustain while gate high, release on fall)
#   - AdsrTriggerSignal: trigger-driven one-shot ADSR with sustain hold time
#
# Output: mono float control signal in [0.0, 1.0]
# Restart behavior on (re)trigger: restart from 0.

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.snippet import Snippet
from pygmu2.extent import Extent
from pygmu2.config import get_sample_rate
from pygmu2.gate_signal import GateSignal
from pygmu2.trigger_signal import TriggerSignal


# -----------------------------
# Internal helpers / state
# -----------------------------

@dataclass
class _EnvState:
    phase: str = "IDLE"      # IDLE | ATTACK | DECAY | SUSTAIN | RELEASE
    pos: int = 0             # sample index within A/D/R phase
    level: float = 0.0       # current level (last rendered sample)
    rel_start: float = 0.0   # starting level for current release
    hold_pos: int = 0        # samples elapsed in sustain-hold (trigger ADSR)


def _sec_to_samps(seconds: float, sr: int) -> int:
    if seconds <= 0:
        return 0
    return int(round(seconds * sr))


def _ramp_progress(pos: int, n: int, total: int) -> np.ndarray:
    """
    Return progress values in (0..1] for n samples starting at phase position `pos`
    for a phase of length `total`. Uses (pos+i+1)/total so last sample hits 1.0.
    """
    if total <= 0:
        return np.ones(n, dtype=np.float32)
    i = np.arange(n, dtype=np.float32)
    return (i + (pos + 1)) / float(total)


# -----------------------------
# Shared ADSR engine (internal)
# -----------------------------

class _AdsrBaseSignal(ProcessingElement):
    """
    Internal ADSR engine.
    Subclasses generate restart/release events and call:
      - self._event_restart()
      - self._event_release()
    and optionally set self._auto_release_remaining for trigger-hold semantics.

    Output is mono float in [0, 1].
    """

    def __init__(
        self,
        attack: float,
        decay: float,
        sustain: float,
        release: float,
        hold: float | None = None,   # only for trigger ADSR (sustain hold duration)
    ):
        sr = int(get_sample_rate())
        self._A = _sec_to_samps(float(attack), sr)
        self._D = _sec_to_samps(float(decay), sr)
        self._R = _sec_to_samps(float(release), sr)
        self._H = _sec_to_samps(float(hold), sr) if hold is not None else None
        self._S = float(sustain)

        if not (0.0 <= self._S <= 1.0):
            raise ValueError("ADSR sustain must be in [0, 1]")

        self._env = _EnvState()

        # For trigger ADSR: counts remaining samples of sustain-hold before auto-release
        # None means "disabled / gate-driven"
        self._auto_release_remaining: int | None = None

    def is_pure(self) -> bool:
        return False

    def channel_count(self) -> int:
        return 1

    def _reset_state(self) -> None:
        self._env = _EnvState()
        self._auto_release_remaining = None

    def _on_start(self) -> None:
        self._reset_state()

    def _on_stop(self) -> None:
        self._reset_state()

    # ---- events ----

    def _event_restart(self) -> None:
        # restart from 0 and enter ATTACK
        self._env.phase = "ATTACK"
        self._env.pos = 0
        self._env.level = 0.0
        self._env.rel_start = 0.0
        self._env.hold_pos = 0
        if self._H is not None:
            self._auto_release_remaining = self._H

    def _event_release(self) -> None:
        # enter RELEASE from current level
        if self._env.phase == "IDLE":
            return
        self._env.phase = "RELEASE"
        self._env.pos = 0
        self._env.rel_start = float(self._env.level)
        self._env.hold_pos = 0
        self._auto_release_remaining = None

    # ---- core evolution ----

    def _render_run(self, n: int, sustain_allowed: bool = True) -> np.ndarray:
        """
        Render n samples continuing the current envelope run.
        If sustain_allowed is False, then SUSTAIN is treated as "start release immediately"
        (used for trigger ADSR after hold expires).
        """
        out = np.zeros((n,), dtype=np.float32)
        idx = 0

        while idx < n:
            phase = self._env.phase

            if phase == "IDLE":
                take = n - idx
                out[idx:idx + take] = 0.0
                self._env.level = 0.0
                idx += take
                continue

            if phase == "ATTACK":
                if self._A == 0:
                    self._env.level = 1.0
                    self._env.phase = "DECAY"
                    self._env.pos = 0
                    continue

                remaining = self._A - self._env.pos
                take = min(n - idx, max(0, remaining))
                if take <= 0:
                    self._env.level = 1.0
                    self._env.phase = "DECAY"
                    self._env.pos = 0
                    continue

                prog = _ramp_progress(self._env.pos, take, self._A)
                seg = prog  # ramp 0->1
                out[idx:idx + take] = seg
                self._env.pos += take
                self._env.level = float(seg[-1])
                idx += take

                if self._env.pos >= self._A:
                    self._env.level = 1.0
                    self._env.phase = "DECAY"
                    self._env.pos = 0
                continue

            if phase == "DECAY":
                if self._D == 0:
                    self._env.level = self._S
                    self._env.phase = "SUSTAIN"
                    self._env.pos = 0
                    self._env.hold_pos = 0
                    continue

                remaining = self._D - self._env.pos
                take = min(n - idx, max(0, remaining))
                if take <= 0:
                    self._env.level = self._S
                    self._env.phase = "SUSTAIN"
                    self._env.pos = 0
                    self._env.hold_pos = 0
                    continue

                prog = _ramp_progress(self._env.pos, take, self._D)
                seg = 1.0 + (self._S - 1.0) * prog
                out[idx:idx + take] = seg
                self._env.pos += take
                self._env.level = float(seg[-1])
                idx += take

                if self._env.pos >= self._D:
                    self._env.level = self._S
                    self._env.phase = "SUSTAIN"
                    self._env.pos = 0
                    self._env.hold_pos = 0
                continue

            if phase == "SUSTAIN":
                if not sustain_allowed:
                    # fall through to release immediately
                    self._env.phase = "RELEASE"
                    self._env.pos = 0
                    self._env.rel_start = float(self._env.level)
                    self._env.hold_pos = 0
                    continue

                take = n - idx
                out[idx:idx + take] = self._S
                self._env.level = self._S
                idx += take
                continue

            if phase == "RELEASE":
                if self._R == 0:
                    take = n - idx
                    out[idx:idx + take] = 0.0
                    self._env.level = 0.0
                    self._env.phase = "IDLE"
                    self._env.pos = 0
                    self._env.rel_start = 0.0
                    self._env.hold_pos = 0
                    idx += take
                    continue

                remaining = self._R - self._env.pos
                take = min(n - idx, max(0, remaining))
                if take <= 0:
                    self._env.level = 0.0
                    self._env.phase = "IDLE"
                    self._env.pos = 0
                    self._env.rel_start = 0.0
                    self._env.hold_pos = 0
                    continue

                prog = _ramp_progress(self._env.pos, take, self._R)
                seg = self._env.rel_start * (1.0 - prog)
                out[idx:idx + take] = seg
                self._env.pos += take
                self._env.level = float(seg[-1])
                idx += take

                if self._env.pos >= self._R:
                    self._env.level = 0.0
                    self._env.phase = "IDLE"
                    self._env.pos = 0
                    self._env.rel_start = 0.0
                    self._env.hold_pos = 0
                continue

            raise RuntimeError(f"Unknown ADSR phase: {phase}")

        return out


# -----------------------------
# Public: Gate-driven ADSR
# -----------------------------

class AdsrGateSignal(_AdsrBaseSignal):
    """
    Gate-driven ADSR.

    - gate is a GateSignal (mono, values 0 or 1).
    - Rising edge: restart from 0 and enter ATTACK.
    - While gate high: progress ATTACK -> DECAY -> SUSTAIN.
    - Falling edge: enter RELEASE from current level.
    """

    def __init__(
        self,
        gate: GateSignal,
        attack: float,
        decay: float,
        sustain: float,
        release: float,
    ):
        super().__init__(attack=attack, decay=decay, sustain=sustain, release=release, hold=None)
        self._gate = gate
        self._prev_gate = 0.0

    def inputs(self) -> list[ProcessingElement]:
        return [self._gate]

    def _compute_extent(self) -> Extent:
        return self._gate.extent()

    def _reset_state(self) -> None:
        super()._reset_state()
        self._prev_gate = 0.0

    def _render(self, start: int, duration: int) -> Snippet:
        gate = self._gate.render(start, duration).data[:, 0]  # 0/1 guaranteed
        out = np.zeros((duration,), dtype=np.float32)

        prev = float(self._prev_gate)
        i = 0
        while i < duration:
            cur = float(gate[i])

            # edge at i
            if prev == 0.0 and cur == 1.0:
                self._event_restart()
            elif prev == 1.0 and cur == 0.0:
                self._event_release()

            # segment until next change in gate
            j = i + 1
            while j < duration and float(gate[j]) == cur:
                j += 1

            # If gate is low and we're in A/D/S, we should be releasing; if gate is high
            # and we're idle, we should start attack. Enforce by injecting events.
            gate_high = (cur == 1.0)
            if gate_high and self._env.phase == "IDLE":
                self._event_restart()
            if (not gate_high) and self._env.phase in ("ATTACK", "DECAY", "SUSTAIN"):
                self._event_release()

            out[i:j] = self._render_run(j - i, sustain_allowed=True)

            prev = cur
            i = j

        self._prev_gate = prev
        return Snippet(start, out.reshape(-1, 1))


# -----------------------------
# Public: Trigger-driven one-shot ADSR
# -----------------------------

class AdsrTriggerSignal(_AdsrBaseSignal):
    """
    Trigger-driven one-shot ADSR with sustain hold.

    - trigger is a TriggerSignal (mono, integer-valued).
    - On trigger > 0: restart from 0 and enter ATTACK.
    - Progress ATTACK -> DECAY -> SUSTAIN for `hold` seconds -> RELEASE -> IDLE.
    - New trigger while active: restart from 0.
    """

    def __init__(
        self,
        trigger: TriggerSignal,
        attack: float,
        decay: float,
        sustain: float,
        hold: float,
        release: float,
    ):
        super().__init__(attack=attack, decay=decay, sustain=sustain, release=release, hold=hold)
        self._trigger = trigger

    def inputs(self) -> list[ProcessingElement]:
        return [self._trigger]

    def _compute_extent(self) -> Extent:
        return self._trigger.extent()

    def _render(self, start: int, duration: int) -> Snippet:
        trig = self._trigger.render(start, duration).data[:, 0]
        out = np.zeros((duration,), dtype=np.float32)

        event_idxs = np.nonzero(trig > 0)[0].tolist()
        i = 0

        for k in event_idxs:
            if k < i:
                continue

            # render up to trigger
            if k > i:
                out[i:k] = self._render_trigger_block(k - i)

            # restart at trigger sample
            self._event_restart()
            out[k:k + 1] = self._render_trigger_block(1)
            i = k + 1

        # tail
        if i < duration:
            out[i:duration] = self._render_trigger_block(duration - i)

        return Snippet(start, out.reshape(-1, 1))

    def _render_trigger_block(self, n: int) -> np.ndarray:
        """
        Render n samples for trigger ADSR, respecting sustain-hold auto-release.
        """
        out = np.zeros((n,), dtype=np.float32)
        idx = 0

        while idx < n:
            # If we're in sustain and hold is enabled, count down hold samples then release.
            if self._env.phase == "SUSTAIN" and self._auto_release_remaining is not None:
                # render min(remaining_hold, remaining_block) at sustain
                take = min(n - idx, self._auto_release_remaining)
                if take > 0:
                    out[idx:idx + take] = self._S
                    self._env.level = self._S
                    self._auto_release_remaining -= take
                    idx += take

                if self._auto_release_remaining <= 0:
                    self._event_release()
                continue

            # Otherwise render using the base evolution for the remainder of this block.
            # Sustain is allowed while hold is running (handled above); when hold expires,
            # we convert sustain into release by calling _event_release().
            take = n - idx
            out[idx:idx + take] = self._render_run(take, sustain_allowed=True)
            idx += take

        return out
