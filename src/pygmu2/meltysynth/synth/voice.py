import math
from enum import IntEnum
from typing import TYPE_CHECKING

from pygmu2.meltysynth.math_utils import HALF_PI, NON_AUDIBLE, SoundFontMath, create_buffer
from pygmu2.meltysynth.model.instrument import InstrumentRegion
from pygmu2.meltysynth.synth.envelope import ModulationEnvelope, VolumeEnvelope
from pygmu2.meltysynth.synth.filter_ import BiQuadFilter
from pygmu2.meltysynth.synth.lfo import Lfo
from pygmu2.meltysynth.synth.oscillator import Oscillator
from pygmu2.meltysynth.synth.channel import Channel
from pygmu2.meltysynth.synth.region_ex import RegionEx
from pygmu2.meltysynth.synth.region_pair import RegionPair

if TYPE_CHECKING:
    from pygmu2.meltysynth.synth.synthesizer import Synthesizer


class VoiceState(IntEnum):
    PLAYING = 0
    RELEASE_REQUESTED = 1
    RELEASED = 2


class Voice:
    def __init__(self, synthesizer: "Synthesizer") -> None:
        self._synthesizer = synthesizer
        self._vol_env = VolumeEnvelope(synthesizer)
        self._mod_env = ModulationEnvelope(synthesizer)
        self._vib_lfo = Lfo(synthesizer)
        self._mod_lfo = Lfo(synthesizer)
        self._oscillator = Oscillator(synthesizer)
        self._filter = BiQuadFilter(synthesizer)
        self._block = create_buffer(synthesizer.block_size)
        self._previous_mix_gain_left = 0
        self._previous_mix_gain_right = 0
        self._current_mix_gain_left = 0
        self._current_mix_gain_right = 0
        self._previous_reverb_send = 0
        self._previous_chorus_send = 0
        self._current_reverb_send = 0
        self._current_chorus_send = 0

    def start(
        self,
        region: RegionPair,
        channel: int,
        key: int,
        velocity: int,
    ) -> None:
        self._exclusive_class = region.exclusive_class
        self._channel = channel
        self._key = key
        self._velocity = velocity

        if velocity > 0:
            sample_attenuation = 0.4 * region.initial_attenuation
            filter_attenuation = 0.5 * region.initial_filter_q
            decibels = (
                2 * SoundFontMath.linear_to_decibels(velocity / 127.0)
                - sample_attenuation
                - filter_attenuation
            )
            self._note_gain = SoundFontMath.decibels_to_linear(decibels)
        else:
            self._note_gain = 0

        self._cutoff = region.initial_filter_cutoff_frequency
        self._resonance = SoundFontMath.decibels_to_linear(
            region.initial_filter_q
        )
        self._vib_lfo_to_pitch = 0.01 * region.vibrato_lfo_to_pitch
        self._mod_lfo_to_pitch = 0.01 * region.modulation_lfo_to_pitch
        self._mod_env_to_pitch = 0.01 * region.modulation_envelope_to_pitch
        self._mod_lfo_to_cutoff = (
            region.modulation_lfo_to_filter_cutoff_frequency
        )
        self._mod_env_to_cutoff = (
            region.modulation_envelope_to_filter_cutoff_frequency
        )
        self._dynamic_cutoff = (
            self._mod_lfo_to_cutoff != 0 or self._mod_env_to_cutoff != 0
        )
        self._mod_lfo_to_volume = region.modulation_lfo_to_volume
        self._dynamic_volume = self._mod_lfo_to_volume > 0.05
        self._instrument_pan = SoundFontMath.clamp(region.pan, -50, 50)
        self._instrument_reverb = 0.01 * region.reverb_effects_send
        self._instrument_chorus = 0.01 * region.chorus_effects_send

        RegionEx.start_volume_envelope(
            self._vol_env, region, key, velocity
        )
        RegionEx.start_modulation_envelope(
            self._mod_env, region, key, velocity
        )
        RegionEx.start_vibrato(self._vib_lfo, region, key, velocity)
        RegionEx.start_modulation(self._mod_lfo, region, key, velocity)
        RegionEx.start_oscillator(
            self._oscillator,
            self._synthesizer.sound_font.wave_data,
            region,
        )
        self._filter.clear_buffer()
        self._filter.set_low_pass_filter(self._cutoff, self._resonance)
        self._smoothed_cutoff = self._cutoff
        self._voice_state = VoiceState.PLAYING
        self._voice_length = 0

    def end(self) -> None:
        if self._voice_state == VoiceState.PLAYING:
            self._voice_state = VoiceState.RELEASE_REQUESTED

    def kill(self) -> None:
        self._note_gain = 0

    def process(self) -> bool:
        if self._note_gain < NON_AUDIBLE:
            return False

        channel_info: Channel = self._synthesizer._channels[self._channel]
        self.release_if_necessary(channel_info)

        if not self._vol_env.process(self._synthesizer.block_size):
            return False

        self._mod_env.process(self._synthesizer.block_size)
        self._vib_lfo.process()
        self._mod_lfo.process()

        vib_pitch_change = (
            0.01 * channel_info.modulation + self._vib_lfo_to_pitch
        ) * self._vib_lfo.value
        mod_pitch_change = (
            self._mod_lfo_to_pitch * self._mod_lfo.value
            + self._mod_env_to_pitch * self._mod_env.value
        )
        channel_pitch_change = channel_info.tune + channel_info.pitch_bend
        pitch = (
            self._key
            + vib_pitch_change
            + mod_pitch_change
            + channel_pitch_change
        )
        if not self._oscillator.process(self._block, pitch):
            return False

        if self._dynamic_cutoff:
            cents = (
                self._mod_lfo_to_cutoff * self._mod_lfo.value
                + self._mod_env_to_cutoff * self._mod_env.value
            )
            factor = SoundFontMath.cents_to_multiplying_factor(cents)
            new_cutoff = factor * self._cutoff
            lower_limit = 0.5 * self._smoothed_cutoff
            upper_limit = 2.0 * self._smoothed_cutoff
            if new_cutoff < lower_limit:
                self._smoothed_cutoff = lower_limit
            elif new_cutoff > upper_limit:
                self._smoothed_cutoff = upper_limit
            else:
                self._smoothed_cutoff = new_cutoff
            self._filter.set_low_pass_filter(
                self._smoothed_cutoff, self._resonance
            )

        self._filter.process(self._block)

        self._previous_mix_gain_left = self._current_mix_gain_left
        self._previous_mix_gain_right = self._current_mix_gain_right
        self._previous_reverb_send = self._current_reverb_send
        self._previous_chorus_send = self._current_chorus_send

        ve = channel_info.volume * channel_info.expression
        channel_gain = ve * ve
        mix_gain = (
            self._note_gain * channel_gain * self._vol_env.value
        )
        if self._dynamic_volume:
            decibels = self._mod_lfo_to_volume * self._mod_lfo.value
            mix_gain *= SoundFontMath.decibels_to_linear(decibels)

        angle = (
            (math.pi / 200.0)
            * (channel_info.pan + self._instrument_pan + 50.0)
        )
        if angle <= 0:
            self._current_mix_gain_left = mix_gain
            self._current_mix_gain_right = 0
        elif angle >= HALF_PI:
            self._current_mix_gain_left = 0
            self._current_mix_gain_right = mix_gain
        else:
            self._current_mix_gain_left = mix_gain * math.cos(angle)
            self._current_mix_gain_right = mix_gain * math.sin(angle)

        self._current_reverb_send = SoundFontMath.clamp(
            channel_info.reverb_send + self._instrument_reverb, 0, 1
        )
        self._current_chorus_send = SoundFontMath.clamp(
            channel_info.chorus_send + self._instrument_chorus, 0, 1
        )

        if self._voice_length == 0:
            self._previous_mix_gain_left = self._current_mix_gain_left
            self._previous_mix_gain_right = self._current_mix_gain_right
            self._previous_reverb_send = self._current_reverb_send
            self._previous_chorus_send = self._current_chorus_send

        self._voice_length += self._synthesizer.block_size
        return True

    def release_if_necessary(self, channel_info: Channel) -> None:
        if self._voice_length < self._synthesizer._minimum_voice_duration:
            return
        if (
            self._voice_state == VoiceState.RELEASE_REQUESTED
            and not channel_info.hold_pedal
        ):
            self._vol_env.release()
            self._mod_env.release()
            self._oscillator.release()
            self._voice_state = VoiceState.RELEASED

    @property
    def priority(self) -> float:
        if self._note_gain < NON_AUDIBLE:
            return 0
        else:
            return self._vol_env.priority

    @property
    def block(self):
        return self._block

    @property
    def previous_mix_gain_left(self) -> float:
        return self._previous_mix_gain_left

    @property
    def previous_mix_gain_right(self) -> float:
        return self._previous_mix_gain_right

    @property
    def current_mix_gain_left(self) -> float:
        return self._current_mix_gain_left

    @property
    def current_mix_gain_right(self) -> float:
        return self._current_mix_gain_right

    @property
    def previous_reverb_send(self) -> float:
        return self._previous_reverb_send

    @property
    def previous_chorus_send(self) -> float:
        return self._previous_chorus_send

    @property
    def current_reverb_send(self) -> float:
        return self._current_reverb_send

    @property
    def current_chorus_send(self) -> float:
        return self._current_chorus_send

    @property
    def exclusive_class(self) -> int:
        return self._exclusive_class

    @property
    def channel(self) -> int:
        return self._channel

    @property
    def key(self) -> int:
        return self._key

    @property
    def velocity(self) -> int:
        return self._velocity

    @property
    def voice_length(self) -> int:
        return self._voice_length
