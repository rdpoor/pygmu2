import itertools
from array import array
from typing import Sequence

from pygmu2.meltysynth.math_utils import SoundFontMath
from pygmu2.meltysynth.model.generator import Generator
from pygmu2.meltysynth.model.instrument_info import InstrumentInfo
from pygmu2.meltysynth.model.sample_header import SampleHeader
from pygmu2.meltysynth.model.types import GeneratorType, LoopMode
from pygmu2.meltysynth.model.zone import Zone


class Instrument:
    def __init__(
        self,
        info: InstrumentInfo,
        zones: Sequence[Zone],
        samples: Sequence[SampleHeader],
    ) -> None:
        self._name = info.name

        zone_count = info.zone_end_index - info.zone_start_index + 1
        if zone_count <= 0:
            raise MeltysynthError(f"The instrument '{info.name}' has no zone.")

        zone_span = zones[
            info.zone_start_index : info.zone_start_index + zone_count
        ]

        self._regions = InstrumentRegion._create(self, zone_span, samples)

    @staticmethod
    def _create(
        infos: Sequence[InstrumentInfo],
        zones: Sequence[Zone],
        samples: Sequence[SampleHeader],
    ) -> Sequence["Instrument"]:
        if len(infos) <= 1:
            raise Exception("No valid instrument was found.")

        count = len(infos) - 1
        instruments: list[Instrument] = []

        for i in range(count):
            instruments.append(Instrument(infos[i], zones, samples))

        return instruments

    @property
    def name(self) -> str:
        return self._name

    @property
    def regions(self) -> Sequence["InstrumentRegion"]:
        return self._regions


class InstrumentRegion:
    def __init__(
        self,
        instrument: Instrument,
        global_zone: Zone,
        local_zone: Zone,
        samples: Sequence[SampleHeader],
    ) -> None:
        self._gs = array("h", itertools.repeat(0, 61))
        self._gs[GeneratorType.INITIAL_FILTER_CUTOFF_FREQUENCY] = 13500
        self._gs[GeneratorType.DELAY_MODULATION_LFO] = -12000
        self._gs[GeneratorType.DELAY_VIBRATO_LFO] = -12000
        self._gs[GeneratorType.DELAY_MODULATION_ENVELOPE] = -12000
        self._gs[GeneratorType.ATTACK_MODULATION_ENVELOPE] = -12000
        self._gs[GeneratorType.HOLD_MODULATION_ENVELOPE] = -12000
        self._gs[GeneratorType.DECAY_MODULATION_ENVELOPE] = -12000
        self._gs[GeneratorType.RELEASE_MODULATION_ENVELOPE] = -12000
        self._gs[GeneratorType.DELAY_VOLUME_ENVELOPE] = -12000
        self._gs[GeneratorType.ATTACK_VOLUME_ENVELOPE] = -12000
        self._gs[GeneratorType.HOLD_VOLUME_ENVELOPE] = -12000
        self._gs[GeneratorType.DECAY_VOLUME_ENVELOPE] = -12000
        self._gs[GeneratorType.RELEASE_VOLUME_ENVELOPE] = -12000
        self._gs[GeneratorType.KEY_RANGE] = 0x7F00
        self._gs[GeneratorType.VELOCITY_RANGE] = 0x7F00
        self._gs[GeneratorType.KEY_NUMBER] = -1
        self._gs[GeneratorType.VELOCITY] = -1
        self._gs[GeneratorType.SCALE_TUNING] = 100
        self._gs[GeneratorType.OVERRIDING_ROOT_KEY] = -1

        for generator in global_zone.generators:
            self._set_parameter(generator)

        for generator in local_zone.generators:
            self._set_parameter(generator)

        sample_id = self._gs[GeneratorType.SAMPLE_ID]
        if not (0 <= sample_id and sample_id < len(samples)):
            raise MeltysynthError(
                f"The instrument '{instrument.name}' contains an invalid sample ID '{sample_id}'."
            )
        self._sample = samples[sample_id]

    @staticmethod
    def _create(
        instrument: Instrument,
        zones: Sequence[Zone],
        samples: Sequence[SampleHeader],
    ) -> Sequence["InstrumentRegion"]:
        if (
            len(zones[0].generators) == 0
            or zones[0].generators[-1].generator_type != GeneratorType.SAMPLE_ID
        ):
            global_zone = zones[0]
            count = len(zones) - 1
            regions: list[InstrumentRegion] = []
            for i in range(count):
                regions.append(
                    InstrumentRegion(instrument, global_zone, zones[i + 1], samples)
                )
            return regions
        else:
            count = len(zones)
            regions = []
            for i in range(count):
                regions.append(
                    InstrumentRegion(
                        instrument, Zone.empty(), zones[i], samples
                    )
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
    def sample(self) -> SampleHeader:
        return self._sample

    @property
    def sample_start(self) -> int:
        return self._sample.start + self.start_address_offset

    @property
    def sample_end(self) -> int:
        return self._sample.end + self.end_address_offset

    @property
    def sample_start_loop(self) -> int:
        return self._sample.start_loop + self.start_loop_address_offset

    @property
    def sample_end_loop(self) -> int:
        return self._sample.end_loop + self.end_loop_address_offset

    @property
    def start_address_offset(self) -> int:
        return (
            32768 * self._gs[GeneratorType.START_ADDRESS_COARSE_OFFSET]
            + self._gs[GeneratorType.START_ADDRESS_OFFSET]
        )

    @property
    def end_address_offset(self) -> int:
        return (
            32768 * self._gs[GeneratorType.END_ADDRESS_COARSE_OFFSET]
            + self._gs[GeneratorType.END_ADDRESS_OFFSET]
        )

    @property
    def start_loop_address_offset(self) -> int:
        return (
            32768 * self._gs[GeneratorType.START_LOOP_ADDRESS_COARSE_OFFSET]
            + self._gs[GeneratorType.START_LOOP_ADDRESS_OFFSET]
        )

    @property
    def end_loop_address_offset(self) -> int:
        return (
            32768 * self._gs[GeneratorType.END_LOOP_ADDRESS_COARSE_OFFSET]
            + self._gs[GeneratorType.END_LOOP_ADDRESS_OFFSET]
        )

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
        return SoundFontMath.cents_to_hertz(
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
        return SoundFontMath.timecents_to_seconds(
            self._gs[GeneratorType.DELAY_MODULATION_LFO]
        )

    @property
    def frequency_modulation_lfo(self) -> float:
        return SoundFontMath.cents_to_hertz(
            self._gs[GeneratorType.FREQUENCY_MODULATION_LFO]
        )

    @property
    def delay_vibrato_lfo(self) -> float:
        return SoundFontMath.timecents_to_seconds(
            self._gs[GeneratorType.DELAY_VIBRATO_LFO]
        )

    @property
    def frequency_vibrato_lfo(self) -> float:
        return SoundFontMath.cents_to_hertz(
            self._gs[GeneratorType.FREQUENCY_VIBRATO_LFO]
        )

    @property
    def delay_modulation_envelope(self) -> float:
        return SoundFontMath.timecents_to_seconds(
            self._gs[GeneratorType.DELAY_MODULATION_ENVELOPE]
        )

    @property
    def attack_modulation_envelope(self) -> float:
        return SoundFontMath.timecents_to_seconds(
            self._gs[GeneratorType.ATTACK_MODULATION_ENVELOPE]
        )

    @property
    def hold_modulation_envelope(self) -> float:
        return SoundFontMath.timecents_to_seconds(
            self._gs[GeneratorType.HOLD_MODULATION_ENVELOPE]
        )

    @property
    def decay_modulation_envelope(self) -> float:
        return SoundFontMath.timecents_to_seconds(
            self._gs[GeneratorType.DECAY_MODULATION_ENVELOPE]
        )

    @property
    def sustain_modulation_envelope(self) -> float:
        return 0.1 * self._gs[GeneratorType.SUSTAIN_MODULATION_ENVELOPE]

    @property
    def release_modulation_envelope(self) -> float:
        return SoundFontMath.timecents_to_seconds(
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
        return SoundFontMath.timecents_to_seconds(
            self._gs[GeneratorType.DELAY_VOLUME_ENVELOPE]
        )

    @property
    def attack_volume_envelope(self) -> float:
        return SoundFontMath.timecents_to_seconds(
            self._gs[GeneratorType.ATTACK_VOLUME_ENVELOPE]
        )

    @property
    def hold_volume_envelope(self) -> float:
        return SoundFontMath.timecents_to_seconds(
            self._gs[GeneratorType.HOLD_VOLUME_ENVELOPE]
        )

    @property
    def decay_volume_envelope(self) -> float:
        return SoundFontMath.timecents_to_seconds(
            self._gs[GeneratorType.DECAY_VOLUME_ENVELOPE]
        )

    @property
    def sustain_volume_envelope(self) -> float:
        return 0.1 * self._gs[GeneratorType.SUSTAIN_VOLUME_ENVELOPE]

    @property
    def release_volume_envelope(self) -> float:
        return SoundFontMath.timecents_to_seconds(
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
        return self._gs[GeneratorType.FINE_TUNE] + self._sample.pitch_correction

    @property
    def sample_modes(self) -> LoopMode:
        return (
            LoopMode(self._gs[GeneratorType.SAMPLE_MODES])
            if self._gs[GeneratorType.SAMPLE_MODES] != 2
            else LoopMode.NO_LOOP
        )

    @property
    def scale_tuning(self) -> int:
        return self._gs[GeneratorType.SCALE_TUNING]

    @property
    def exclusive_class(self) -> int:
        return self._gs[GeneratorType.EXCLUSIVE_CLASS]

    @property
    def root_key(self) -> int:
        return (
            self._gs[GeneratorType.OVERRIDING_ROOT_KEY]
            if self._gs[GeneratorType.OVERRIDING_ROOT_KEY] != -1
            else self._sample.original_pitch
        )
