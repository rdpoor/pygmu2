from typing import Sequence

from pygmu2.meltysynth.math_utils import SoundFontMath
from pygmu2.meltysynth.model.generator import Generator
from pygmu2.meltysynth.model.instrument import Instrument
from pygmu2.meltysynth.model.preset_info import PresetInfo
from pygmu2.meltysynth.model.types import GeneratorType
from pygmu2.meltysynth.model.zone import Zone


class Preset:
    def __init__(
        self,
        info: PresetInfo,
        zones: Sequence[Zone],
        instruments: Sequence[Instrument],
    ) -> None:
        self._name = info.name
        self._patch_number = info.patch_number
        self._bank_number = info.bank_number
        self._library = info.library
        self._genre = info.genre
        self._morphology = info.morphology

        zone_count = info.zone_end_index - info.zone_start_index + 1
        if zone_count <= 0:
            raise Exception("The preset '" + info.name + "' has no zone.")

        zone_span = zones[
            info.zone_start_index : info.zone_start_index + zone_count
        ]

        self._regions = PresetRegion._create(self, zone_span, instruments)

    @staticmethod
    def _create(
        infos: Sequence[PresetInfo],
        zones: Sequence[Zone],
        instruments: Sequence[Instrument],
    ) -> Sequence["Preset"]:
        if len(infos) <= 1:
            raise Exception("No valid preset was found.")

        count = len(infos) - 1
        presets: list[Preset] = []

        for i in range(count):
            presets.append(Preset(infos[i], zones, instruments))

        return presets

    @property
    def name(self) -> str:
        return self._name

    @property
    def patch_number(self) -> int:
        return self._patch_number

    @property
    def bank_number(self) -> int:
        return self._bank_number

    @property
    def library(self) -> int:
        return self._library

    @property
    def genre(self) -> int:
        return self._genre

    @property
    def morphology(self) -> int:
        return self._morphology

    @property
    def regions(self) -> Sequence["PresetRegion"]:
        return self._regions


class PresetRegion:
    def __init__(
        self,
        preset: Preset,
        global_zone: Zone,
        local_zone: Zone,
        instruments: Sequence[Instrument],
    ) -> None:
        from array import array
        import itertools

        self._gs = array("h", itertools.repeat(0, 61))
        self._gs[GeneratorType.KEY_RANGE] = 0x7F00
        self._gs[GeneratorType.VELOCITY_RANGE] = 0x7F00

        for generator in global_zone.generators:
            self._set_parameter(generator)

        for generator in local_zone.generators:
            self._set_parameter(generator)

        id = self._gs[GeneratorType.INSTRUMENT]
        if not (0 <= id and id < len(instruments)):
            raise Exception(
                "The preset '"
                + preset.name
                + "' contains an invalid instrument ID '"
                + str(id)
                + "'."
            )
        self._instrument = instruments[id]

    @staticmethod
    def _create(
        preset: Preset,
        zones: Sequence[Zone],
        instruments: Sequence[Instrument],
    ) -> Sequence["PresetRegion"]:
        if (
            len(zones[0].generators) == 0
            or zones[0].generators[-1].generator_type != GeneratorType.INSTRUMENT
        ):
            global_zone = zones[0]
            count = len(zones) - 1
            regions: list[PresetRegion] = []
            for i in range(count):
                regions.append(
                    PresetRegion(preset, global_zone, zones[i + 1], instruments)
                )
            return regions
        else:
            count = len(zones)
            regions = []
            for i in range(count):
                regions.append(
                    PresetRegion(preset, Zone.empty(), zones[i], instruments)
                )
            return regions

    def _set_parameter(self, generator: Generator) -> None:
        index = int(generator.generator_type)
        if 0 <= index and index < len(self._gs):
            self._gs[index] = generator.value

    def contains(self, key: int, velocity: int) -> bool:
        contains_key = self.key_range_start <= key and key <= self.key_range_end
        contains_velocity = (
            self.velocity_range_start <= velocity
            and velocity <= self.velocity_range_end
        )
        return contains_key and contains_velocity

    @property
    def instrument(self) -> Instrument:
        return self._instrument

    @property
    def modulation_lfo_to_pitch(self) -> int:
        return self._gs[GeneratorType.MODULATION_LFO_TO_PITCH]

    @property
    def vibrato_lfo_to_pitch(self) -> int:
        return self._gs[GeneratorType.VIBRATO_LFO_TO_PITCH]

    @property
    def modulation_envelope_to_pitch(self) -> int:
        return self._gs[GeneratorType.MODULATION_ENVELOPE_TO_PITCH]

    @property
    def initial_filter_cutoff_frequency(self) -> float:
        return SoundFontMath.cents_to_multiplying_factor(
            self._gs[GeneratorType.INITIAL_FILTER_CUTOFF_FREQUENCY]
        )

    @property
    def initial_filter_q(self) -> float:
        return 0.1 * self._gs[GeneratorType.INITIAL_FILTER_Q]

    @property
    def modulation_lfo_to_filter_cutoff_frequency(self) -> int:
        return self._gs[GeneratorType.MODULATION_LFO_TO_FILTER_CUTOFF_FREQUENCY]

    @property
    def modulation_envelope_to_filter_cutoff_frequency(self) -> int:
        return self._gs[
            GeneratorType.MODULATION_ENVELOPE_TO_FILTER_CUTOFF_FREQUENCY
        ]

    @property
    def modulation_lfo_to_volume(self) -> float:
        return 0.1 * self._gs[GeneratorType.MODULATION_LFO_TO_VOLUME]

    @property
    def chorus_effects_send(self) -> float:
        return 0.1 * self._gs[GeneratorType.CHORUS_EFFECTS_SEND]

    @property
    def reverb_effects_send(self) -> float:
        return 0.1 * self._gs[GeneratorType.REVERB_EFFECTS_SEND]

    @property
    def pan(self) -> float:
        return 0.1 * self._gs[GeneratorType.PAN]

    @property
    def delay_modulation_lfo(self) -> float:
        return SoundFontMath.cents_to_multiplying_factor(
            self._gs[GeneratorType.DELAY_MODULATION_LFO]
        )

    @property
    def frequency_modulation_lfo(self) -> float:
        return SoundFontMath.cents_to_multiplying_factor(
            self._gs[GeneratorType.FREQUENCY_MODULATION_LFO]
        )

    @property
    def delay_vibrato_lfo(self) -> float:
        return SoundFontMath.cents_to_multiplying_factor(
            self._gs[GeneratorType.DELAY_VIBRATO_LFO]
        )

    @property
    def frequency_vibrato_lfo(self) -> float:
        return SoundFontMath.cents_to_multiplying_factor(
            self._gs[GeneratorType.FREQUENCY_VIBRATO_LFO]
        )

    @property
    def delay_modulation_envelope(self) -> float:
        return SoundFontMath.cents_to_multiplying_factor(
            self._gs[GeneratorType.DELAY_MODULATION_ENVELOPE]
        )

    @property
    def attack_modulation_envelope(self) -> float:
        return SoundFontMath.cents_to_multiplying_factor(
            self._gs[GeneratorType.ATTACK_MODULATION_ENVELOPE]
        )

    @property
    def hold_modulation_envelope(self) -> float:
        return SoundFontMath.cents_to_multiplying_factor(
            self._gs[GeneratorType.HOLD_MODULATION_ENVELOPE]
        )

    @property
    def decay_modulation_envelope(self) -> float:
        return SoundFontMath.cents_to_multiplying_factor(
            self._gs[GeneratorType.DECAY_MODULATION_ENVELOPE]
        )

    @property
    def sustain_modulation_envelope(self) -> float:
        return 0.1 * self._gs[GeneratorType.SUSTAIN_MODULATION_ENVELOPE]

    @property
    def release_modulation_envelope(self) -> float:
        return SoundFontMath.cents_to_multiplying_factor(
            self._gs[GeneratorType.RELEASE_MODULATION_ENVELOPE]
        )

    @property
    def key_number_to_modulation_envelope_hold(self) -> int:
        return self._gs[GeneratorType.KEY_NUMBER_TO_MODULATION_ENVELOPE_HOLD]

    @property
    def key_number_to_modulation_envelope_decay(self) -> int:
        return self._gs[GeneratorType.KEY_NUMBER_TO_MODULATION_ENVELOPE_DECAY]

    @property
    def delay_volume_envelope(self) -> float:
        return SoundFontMath.cents_to_multiplying_factor(
            self._gs[GeneratorType.DELAY_VOLUME_ENVELOPE]
        )

    @property
    def attack_volume_envelope(self) -> float:
        return SoundFontMath.cents_to_multiplying_factor(
            self._gs[GeneratorType.ATTACK_VOLUME_ENVELOPE]
        )

    @property
    def hold_volume_envelope(self) -> float:
        return SoundFontMath.cents_to_multiplying_factor(
            self._gs[GeneratorType.HOLD_VOLUME_ENVELOPE]
        )

    @property
    def decay_volume_envelope(self) -> float:
        return SoundFontMath.cents_to_multiplying_factor(
            self._gs[GeneratorType.DECAY_VOLUME_ENVELOPE]
        )

    @property
    def sustain_volume_envelope(self) -> float:
        return 0.1 * self._gs[GeneratorType.SUSTAIN_VOLUME_ENVELOPE]

    @property
    def release_volume_envelope(self) -> float:
        return SoundFontMath.cents_to_multiplying_factor(
            self._gs[GeneratorType.RELEASE_VOLUME_ENVELOPE]
        )

    @property
    def key_number_to_volume_envelope_hold(self) -> int:
        return self._gs[GeneratorType.KEY_NUMBER_TO_VOLUME_ENVELOPE_HOLD]

    @property
    def key_number_to_volume_envelope_decay(self) -> int:
        return self._gs[GeneratorType.KEY_NUMBER_TO_VOLUME_ENVELOPE_DECAY]

    @property
    def key_range_start(self) -> int:
        return self._gs[GeneratorType.KEY_RANGE] & 0xFF

    @property
    def key_range_end(self) -> int:
        return (self._gs[GeneratorType.KEY_RANGE] >> 8) & 0xFF

    @property
    def velocity_range_start(self) -> int:
        return self._gs[GeneratorType.VELOCITY_RANGE] & 0xFF

    @property
    def velocity_range_end(self) -> int:
        return (self._gs[GeneratorType.VELOCITY_RANGE] >> 8) & 0xFF

    @property
    def initial_attenuation(self) -> float:
        return 0.1 * self._gs[GeneratorType.INITIAL_ATTENUATION]

    @property
    def coarse_tune(self) -> int:
        return self._gs[GeneratorType.COARSE_TUNE]

    @property
    def fine_tune(self) -> int:
        return self._gs[GeneratorType.FINE_TUNE]

    @property
    def scale_tuning(self) -> int:
        return self._gs[GeneratorType.SCALE_TUNING]
