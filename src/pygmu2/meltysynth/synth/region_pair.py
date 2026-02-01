from pygmu2.meltysynth.math_utils import SoundFontMath
from pygmu2.meltysynth.model.instrument import InstrumentRegion
from pygmu2.meltysynth.model.preset import PresetRegion
from pygmu2.meltysynth.model.types import GeneratorType


class RegionPair:
    def __init__(
        self, preset: PresetRegion, instrument: InstrumentRegion
    ) -> None:
        self._preset = preset
        self._instrument = instrument

    def get_value(self, generator_type: GeneratorType) -> int:
        return self._preset._gs[generator_type] + self._instrument._gs[
            generator_type
        ]

    @property
    def preset(self) -> PresetRegion:
        return self._preset

    @property
    def instrument(self) -> InstrumentRegion:
        return self._instrument

    @property
    def sample_start(self) -> int:
        return self._instrument.sample_start

    @property
    def sample_end(self) -> int:
        return self._instrument.sample_end

    @property
    def sample_start_loop(self) -> int:
        return self._instrument.sample_start_loop

    @property
    def sample_end_loop(self) -> int:
        return self._instrument.sample_end_loop

    @property
    def start_address_offset(self) -> int:
        return self._instrument.start_address_offset

    @property
    def end_address_offset(self) -> int:
        return self._instrument.end_address_offset

    @property
    def start_loop_address_offset(self) -> int:
        return self._instrument.start_loop_address_offset

    @property
    def end_loop_address_offset(self) -> int:
        return self._instrument.end_loop_address_offset

    @property
    def modulation_lfo_to_pitch(self) -> int:
        return self.get_value(GeneratorType.MODULATION_LFO_TO_PITCH)

    @property
    def vibrato_lfo_to_pitch(self) -> int:
        return self.get_value(GeneratorType.VIBRATO_LFO_TO_PITCH)

    @property
    def modulation_envelope_to_pitch(self) -> int:
        return self.get_value(GeneratorType.MODULATION_ENVELOPE_TO_PITCH)

    @property
    def initial_filter_cutoff_frequency(self) -> float:
        return SoundFontMath.cents_to_hertz(
            self.get_value(GeneratorType.INITIAL_FILTER_CUTOFF_FREQUENCY)
        )

    @property
    def initial_filter_q(self) -> float:
        return 0.1 * self.get_value(GeneratorType.INITIAL_FILTER_Q)

    @property
    def modulation_lfo_to_filter_cutoff_frequency(self) -> int:
        return self.get_value(
            GeneratorType.MODULATION_LFO_TO_FILTER_CUTOFF_FREQUENCY
        )

    @property
    def modulation_envelope_to_filter_cutoff_frequency(self) -> int:
        return self.get_value(
            GeneratorType.MODULATION_ENVELOPE_TO_FILTER_CUTOFF_FREQUENCY
        )

    @property
    def modulation_lfo_to_volume(self) -> float:
        return 0.1 * self.get_value(GeneratorType.MODULATION_LFO_TO_VOLUME)

    @property
    def chorus_effects_send(self) -> float:
        return 0.1 * self.get_value(GeneratorType.CHORUS_EFFECTS_SEND)

    @property
    def reverb_effects_send(self) -> float:
        return 0.1 * self.get_value(GeneratorType.REVERB_EFFECTS_SEND)

    @property
    def pan(self) -> float:
        return 0.1 * self.get_value(GeneratorType.PAN)

    @property
    def delay_modulation_lfo(self) -> float:
        return SoundFontMath.timecents_to_seconds(
            self.get_value(GeneratorType.DELAY_MODULATION_LFO)
        )

    @property
    def frequency_modulation_lfo(self) -> float:
        return SoundFontMath.cents_to_hertz(
            self.get_value(GeneratorType.FREQUENCY_MODULATION_LFO)
        )

    @property
    def delay_vibrato_lfo(self) -> float:
        return SoundFontMath.timecents_to_seconds(
            self.get_value(GeneratorType.DELAY_VIBRATO_LFO)
        )

    @property
    def frequency_vibrato_lfo(self) -> float:
        return SoundFontMath.cents_to_hertz(
            self.get_value(GeneratorType.FREQUENCY_VIBRATO_LFO)
        )

    @property
    def delay_modulation_envelope(self) -> float:
        return SoundFontMath.timecents_to_seconds(
            self.get_value(GeneratorType.DELAY_MODULATION_ENVELOPE)
        )

    @property
    def attack_modulation_envelope(self) -> float:
        return SoundFontMath.timecents_to_seconds(
            self.get_value(GeneratorType.ATTACK_MODULATION_ENVELOPE)
        )

    @property
    def hold_modulation_envelope(self) -> float:
        return SoundFontMath.timecents_to_seconds(
            self.get_value(GeneratorType.HOLD_MODULATION_ENVELOPE)
        )

    @property
    def decay_modulation_envelope(self) -> float:
        return SoundFontMath.timecents_to_seconds(
            self.get_value(GeneratorType.DECAY_MODULATION_ENVELOPE)
        )

    @property
    def sustain_modulation_envelope(self) -> float:
        return 0.1 * self.get_value(GeneratorType.SUSTAIN_MODULATION_ENVELOPE)

    @property
    def release_modulation_envelope(self) -> float:
        return SoundFontMath.timecents_to_seconds(
            self.get_value(GeneratorType.RELEASE_MODULATION_ENVELOPE)
        )

    @property
    def key_number_to_modulation_envelope_hold(self) -> int:
        return self.get_value(
            GeneratorType.KEY_NUMBER_TO_MODULATION_ENVELOPE_HOLD
        )

    @property
    def key_number_to_modulation_envelope_decay(self) -> int:
        return self.get_value(
            GeneratorType.KEY_NUMBER_TO_MODULATION_ENVELOPE_DECAY
        )

    @property
    def delay_volume_envelope(self) -> float:
        return SoundFontMath.timecents_to_seconds(
            self.get_value(GeneratorType.DELAY_VOLUME_ENVELOPE)
        )

    @property
    def attack_volume_envelope(self) -> float:
        return SoundFontMath.timecents_to_seconds(
            self.get_value(GeneratorType.ATTACK_VOLUME_ENVELOPE)
        )

    @property
    def hold_volume_envelope(self) -> float:
        return SoundFontMath.timecents_to_seconds(
            self.get_value(GeneratorType.HOLD_VOLUME_ENVELOPE)
        )

    @property
    def decay_volume_envelope(self) -> float:
        return SoundFontMath.timecents_to_seconds(
            self.get_value(GeneratorType.DECAY_VOLUME_ENVELOPE)
        )

    @property
    def sustain_volume_envelope(self) -> float:
        return 0.1 * self.get_value(GeneratorType.SUSTAIN_VOLUME_ENVELOPE)

    @property
    def release_volume_envelope(self) -> float:
        return SoundFontMath.timecents_to_seconds(
            self.get_value(GeneratorType.RELEASE_VOLUME_ENVELOPE)
        )

    @property
    def key_number_to_volume_envelope_hold(self) -> int:
        return self.get_value(GeneratorType.KEY_NUMBER_TO_VOLUME_ENVELOPE_HOLD)

    @property
    def key_number_to_volume_envelope_decay(self) -> int:
        return self.get_value(
            GeneratorType.KEY_NUMBER_TO_VOLUME_ENVELOPE_DECAY
        )

    @property
    def initial_attenuation(self) -> float:
        return 0.1 * self.get_value(GeneratorType.INITIAL_ATTENUATION)

    @property
    def coarse_tune(self) -> int:
        return self.get_value(GeneratorType.COARSE_TUNE)

    @property
    def fine_tune(self) -> int:
        return (
            self.get_value(GeneratorType.FINE_TUNE)
            + self._instrument.sample.pitch_correction
        )

    @property
    def sample_modes(self):
        from pygmu2.meltysynth.model.types import LoopMode

        return self._instrument.sample_modes

    @property
    def scale_tuning(self) -> int:
        return self.get_value(GeneratorType.SCALE_TUNING)

    @property
    def exclusive_class(self) -> int:
        return self._instrument.exclusive_class

    @property
    def root_key(self) -> int:
        return self._instrument.root_key
