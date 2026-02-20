"""
adsr_eg.py  ADSR demos using the new GateSignal / TriggerSignal ADSR classes.

Usage:
  uv run python examples/adsr_eg.py
  uv run python examples/adsr_eg.py 1
  uv run python examples/adsr_eg.py a
"""

from __future__ import annotations

import sys
from typing import Callable
import numpy as np

import pygmu2 as pg

pg.set_sample_rate(44100)
SR = 44100


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def seconds(s: float) -> int:
    return int(round(float(s) * SR))


def lin_map(lo: float, hi: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a TransformPE-friendly mapper f(arr)->arr that maps [0..1] to [lo..hi].
    """
    lo = float(lo)
    hi = float(hi)
    span = hi - lo

    def _f(arr: np.ndarray) -> np.ndarray:
        return lo + span * arr

    return _f


# ------------------------------------------------------------------------------
# Demos
# ------------------------------------------------------------------------------

REPEAT_HZ = 1.0
ATTACK_SEC = 0.01
DECAY_SEC = 0.15
HOLD_SEC = 0.5
SUSTAIN_LEVEL = 0.6
RELEASE_SEC = 0.25

REPEAT_SEC = 1.0 / REPEAT_HZ
GATE_SEC = ATTACK_SEC + DECAY_SEC + HOLD_SEC

def demo_gate_adsr_vca():
    """
    PeriodicGate -> AdsrGateSignal -> GainPE(SuperSaw)
    """
    print("Gate ADSR: VCA on SuperSaw")
    print("--------------------------")

    duration_s = 8.0

    if GATE_SEC + RELEASE_SEC > REPEAT_SEC:
        print(f'warning: REPEAT_HZ is too high for ADSR to complete')

    # An on-off gate to simulate a note repeating every 1/REPEAT_HZ seconds,
    # held for GATE_SEC each time.
    # Note: duty_cycle = gate_sec / repeat_sec = gate_sec * repeat_hz
    gate = pg.PeriodicGate(
        frequency=REPEAT_HZ,
        duty_cycle=GATE_SEC / REPEAT_SEC,
    )

    # When the gate signal goes high, the ADSR sequences through Attack => 
    # Decay before settling on the Sustain level.  It holds Sustain as long
    # as the gate signal is high.  When the gate signal drops, the ADSR starts
    # the Release phase.
    env = pg.AdsrGatedPE(
        gate=gate,
        attack_time=ATTACK_SEC,
        decay_time=DECAY_SEC,
        sustain_level=SUSTAIN_LEVEL,
        release_time=RELEASE_SEC,
    )

    # We want the SuperSaw to restart at the same time as the ADSR.  For this,
    # we need a trigger rather than a gate.  
    # 
    # TODO: Implement EdgeSignal that converts a gate signal into a trigger
    # signal.  But for now, use a PeriodicTrigger running at the same speed as
    # the PeriodicGate.
    trigger = pg.PeriodicTrigger(
        hz=REPEAT_HZ,
    )

    # SuperSaw makes a nice fat synth sound.
    saw = pg.SuperSawPE(
        frequency=110.0,     # A2
        voices=7,
        detune_cents=12.0,
    )
    retriggered_saw = pg.TriggerRestartPE(trigger=trigger, src=saw)

    # Apply ADSR envelope as time-varying gain (tremolo/VCA)
    vca = pg.GainPE(retriggered_saw, gain=env)

    # Keep levels sane
    out = pg.GainPE(vca, gain=0.25)

    out = pg.CropPE(out, 0, seconds(duration_s))
    pg.play(out, SR)


def demo_trigger_adsr_vca():
    """
    PeriodicTrigger -> AdsrTriggerSignal -> GainPE(SuperSaw)
    """
    print("Trigger ADSR: VCA on SuperSaw")
    print("--------------------------")

    duration_s = 8.0

    if GATE_SEC + RELEASE_SEC > REPEAT_SEC:
        print(f'warning: REPEAT_HZ is too high for ADSR to complete')

    trigger = pg.PeriodicTrigger(
        hz=REPEAT_HZ,
    )

    env = pg.AdsrTriggeredPE(
        trigger=trigger,
        attack_time=ATTACK_SEC,
        decay_time=DECAY_SEC,
        sustain_level=SUSTAIN_LEVEL,
        sustain_time=HOLD_SEC,
        release_time=RELEASE_SEC,
    )

    saw = pg.SuperSawPE(
        frequency=110.0,     # A2
        voices=7,
        detune_cents=12.0,
    )
    retriggered_saw = pg.TriggerRestartPE(trigger=trigger, src=saw)

    # Apply ADSR envelope as time-varying gain (tremolo/VCA)
    vca = pg.GainPE(retriggered_saw, gain=env)

    # Keep levels sane
    out = pg.GainPE(vca, gain=0.25)

    out = pg.CropPE(out, 0, seconds(duration_s))
    pg.play(out, SR)


def demo_trigger_adsr_filter_sweep():
    """
    PeriodicTrigger -> AdsrTriggerSignal -> TransformPE -> LadderPE cutoff
    """
    print("Trigger ADSR: Resonant filter sweep")
    print("-----------------------------------")

    duration_s = 10.0

    # A periodic trigger to launch one-shot envelopes
    trigger = pg.PeriodicTrigger(hz=0.5, phase=0.0, amplitude=1)

    # One-shot envelope (A, D, sustain hold, R)
    sweep_env = pg.AdsrTriggeredPE(
        trigger=trigger,
        attack_time=0.4,
        decay_time=0.4,
        sustain_level=1.0,
        sustain_time=0.30,
        release_time=0.60,
    )

    # Map envelope [0..1] -> cutoff range (Hz)
    cutoff = pg.TransformPE(sweep_env, lin_map(200.0, 6000.0))

    src = pg.SuperSawPE(
        frequency=55.0,      # A1
        voices=9,
        detune_cents=18.0,
    )
    src = pg.GainPE(src, gain=0.25)

    # Resonant low-pass sweep using BiquadPE
    flt = pg.BiquadPE(
        src,                    # or vca in the dual-ADSR demo
        frequency=cutoff,       # modulated cutoff (PE)
        q=8.0,                  # resonance; tweak 2..12
    )

    out = pg.CropPE(flt, 0, seconds(duration_s))
    pg.play(out, SR)


def demo_dual_adsr_vca_and_filter():
    """
    Two ADSRs:
      - Gate ADSR gates the amplitude (VCA)
      - Trigger ADSR sweeps filter cutoff
    """
    print("Dual ADSR: VCA + filter sweep")
    print("-----------------------------")

    duration_s = 12.0

    # Gate pattern: controls loudness and note "hold"
    gate = pg.PeriodicGate(
        frequency=0.5,
        duty_cycle=0.35,
        phase=0.0,
    )

    amp_env = pg.AdsrGatedPE(
        gate=gate,
        attack_time=0.005,
        decay_time=0.10,
        sustain_level=0.5,
        release_time=0.20,
    )

    # Trigger pattern: independent "wah" sweeps
    trig = pg.PeriodicTrigger(hz=0.5, phase=0.0, amplitude=1)

    filt_env = pg.AdsrTriggeredPE(
        trigger=trig,
        attack_time=0.01,
        decay_time=0.18,
        sustain_level=1.0,
        sustain_time=0.08,
        release_time=0.55,
    )

    cutoff = pg.TransformPE(filt_env, lin_map(250.0, 8000.0))

    src = pg.SuperSawPE(
        frequency=110.0,
        voices=7,
        detune_cents=10.0,
    )

    # VCA (gate ADSR)
    vca = pg.GainPE(src, gain=amp_env)

    # Resonant low-pass sweep using BiquadPE
    flt = pg.BiquadPE(
        vca,                    # or vca in the dual-ADSR demo
        frequency=cutoff,       # modulated cutoff (PE)
        q=8.0,                  # resonance; tweak 2..12
    )

    out = pg.GainPE(flt, gain=0.25)
    out = pg.CropPE(out, 0, seconds(duration_s))
    pg.play(out, SR)


DEMOS = {
    "Gate ADSR: VCA on SuperSaw": demo_gate_adsr_vca,
    "Trigger ADSR: VCA on SuperSaw": demo_trigger_adsr_vca,
    "Trigger ADSR: resonant cutoff sweep": demo_trigger_adsr_filter_sweep,
    "Dual ADSR: VCA + filter sweep": demo_dual_adsr_vca_and_filter,
}


# ------------------------------------------------------------------------------
# Main (matches examples/00_template_eg.py structure)
# ------------------------------------------------------------------------------

def resolve_choice(choice: str):
    """Return (name, fn) on valid choice, (None, None) otherwise."""
    item_list = list(DEMOS.items())

    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(item_list):
            return item_list[idx - 1]
        return None, None

    if choice in DEMOS:
        return choice, DEMOS[choice]

    return None, None


def print_menu():
    names = list(DEMOS.keys())
    print("Available demos:")
    for i, name in enumerate(names, start=1):
        print(f" {i}: {name}")
    print(" ?: show list")
    print(" a: run all")
    print(" q: quit")


def choose_and_play():
    """Interactive chooser loop."""
    while True:
        choice = input("Select demo (name or number): ").strip()
        if choice.lower() == "q":
            break
        if choice.lower() == "a":
            for fn in DEMOS.values():
                fn()
            continue
        if choice == "?":
            print_menu()
            continue

        _name, fn = resolve_choice(choice)
        if fn is not None:
            fn()
        else:
            print(f"unrecognized choice {choice}, '?' to see choices")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
        if choice.lower() == "a":
            for fn in DEMOS.values():
                fn()
            raise SystemExit(0)

        _name, fn = resolve_choice(choice)
        if fn is not None:
            fn()
        else:
            print(f"Invalid choice '{choice}'")
    else:
        print_menu()
        choose_and_play()
