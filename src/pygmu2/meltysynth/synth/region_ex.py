from collections.abc import Sequence
from typing import TYPE_CHECKING

from pygmu2.meltysynth.math_utils import SoundFontMath
from pygmu2.meltysynth.synth.oscillator import Oscillator
from pygmu2.meltysynth.synth.envelope import (
    ModulationEnvelope,
    VolumeEnvelope,
)
from pygmu2.meltysynth.synth.lfo import Lfo
from pygmu2.meltysynth.synth.region_pair import RegionPair

if TYPE_CHECKING:
    pass


class RegionEx:
    @staticmethod
    def start_oscillator(
        oscillator: Oscillator,
        data: Sequence[float],
        region: RegionPair,
    ) -> None:
        sample_rate = region.instrument.sample.sample_rate
        loop_mode = region.sample_modes
        start = region.sample_start
        end = region.sample_end
        start_loop = region.sample_start_loop
        end_loop = region.sample_end_loop
        root_key = region.root_key
        coarse_tune = region.coarse_tune
        fine_tune = region.fine_tune
        scale_tuning = region.scale_tuning

        oscillator.start(
            data,
            loop_mode,
            sample_rate,
            start,
            end,
            start_loop,
            end_loop,
            root_key,
            coarse_tune,
            fine_tune,
            scale_tuning,
        )

    @staticmethod
    def start_volume_envelope(
        envelope: VolumeEnvelope, region: RegionPair, key: int, velocity: int
    ) -> None:
        delay = region.delay_volume_envelope
        attack = region.attack_volume_envelope
        hold = (
            region.hold_volume_envelope
            * SoundFontMath.key_number_to_multiplying_factor(
                region.key_number_to_volume_envelope_hold, key
            )
        )
        decay = (
            region.decay_volume_envelope
            * SoundFontMath.key_number_to_multiplying_factor(
                region.key_number_to_volume_envelope_decay, key
            )
        )
        sustain = SoundFontMath.decibels_to_linear(
            -region.sustain_volume_envelope
        )
        release = max(region.release_volume_envelope, 0.01)

        envelope.start(delay, attack, hold, decay, sustain, release)

    @staticmethod
    def start_modulation_envelope(
        envelope: ModulationEnvelope,
        region: RegionPair,
        key: int,
        velocity: int,
    ) -> None:
        delay = region.delay_modulation_envelope
        attack = region.attack_modulation_envelope * (
            (145 - velocity) / 144.0
        )
        hold = (
            region.hold_modulation_envelope
            * SoundFontMath.key_number_to_multiplying_factor(
                region.key_number_to_modulation_envelope_hold, key
            )
        )
        decay = (
            region.decay_modulation_envelope
            * SoundFontMath.key_number_to_multiplying_factor(
                region.key_number_to_modulation_envelope_decay, key
            )
        )
        sustain = 1.0 - region.sustain_modulation_envelope / 100.0
        release = region.release_modulation_envelope

        envelope.start(delay, attack, hold, decay, sustain, release)

    @staticmethod
    def start_vibrato(
        lfo: Lfo, region: RegionPair, key: int, velocity: int
    ) -> None:
        lfo.start(region.delay_vibrato_lfo, region.frequency_vibrato_lfo)

    @staticmethod
    def start_modulation(
        lfo: Lfo, region: RegionPair, key: int, velocity: int
    ) -> None:
        lfo.start(region.delay_modulation_lfo, region.frequency_modulation_lfo)
