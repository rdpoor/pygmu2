from collections.abc import MutableSequence, Sequence
from typing import Optional

from pygmu2.meltysynth.math_utils import ArrayMath, SoundFontMath, create_buffer
from pygmu2.meltysynth.model.instrument import InstrumentRegion
from pygmu2.meltysynth.model.preset import Preset
from pygmu2.meltysynth.model.soundfont import SoundFont
from pygmu2.meltysynth.synth.channel import Channel
from pygmu2.meltysynth.synth.region_pair import RegionPair
from pygmu2.meltysynth.synth.settings import SynthesizerSettings
from pygmu2.meltysynth.synth.voice_collection import VoiceCollection


class Synthesizer:
    _CHANNEL_COUNT = 16
    _PERCUSSION_CHANNEL = 9
    _INITIAL_MIN_PRESET_ID = 10**10

    def __init__(
        self, sound_font: SoundFont, settings: SynthesizerSettings
    ) -> None:
        self._sound_font = sound_font
        self._sample_rate = settings.sample_rate
        self._block_size = settings.block_size
        self._maximum_polyphony = settings.maximum_polyphony
        self._enable_reverb_and_chorus = settings.enable_reverb_and_chorus
        self._minimum_voice_duration = self._sample_rate // 500
        self._preset_lookup: dict[int, Preset] = {}
        min_preset_id = Synthesizer._INITIAL_MIN_PRESET_ID
        for preset in self._sound_font.presets:
            preset_id = (preset.bank_number << 16) | preset.patch_number
            self._preset_lookup[preset_id] = preset
            if preset_id < min_preset_id:
                self._default_preset = preset
                min_preset_id = preset_id

        self._channels: list[Channel] = []
        for i in range(Synthesizer._CHANNEL_COUNT):
            self._channels.append(
                Channel(self, i == Synthesizer._PERCUSSION_CHANNEL)
            )
        self._voices = VoiceCollection(self, self._maximum_polyphony)
        self._block_left = create_buffer(self._block_size)
        self._block_right = create_buffer(self._block_size)
        self._inverse_block_size = 1.0 / self._block_size
        self._block_read = self._block_size
        self._master_volume = 0.5

    def process_midi_message(
        self, channel: int, command: int, data1: int, data2: int
    ) -> None:
        if not (0 <= channel and channel < len(self._channels)):
            return
        channel_info = self._channels[channel]

        match command:
            case 0x80:
                self.note_off(channel, data1)
            case 0x90:
                self.note_on(channel, data1, data2)
            case 0xB0:
                match data1:
                    case 0x00:
                        channel_info.set_bank(data2)
                    case 0x01:
                        channel_info.set_modulation_coarse(data2)
                    case 0x21:
                        channel_info.set_modulation_fine(data2)
                    case 0x06:
                        channel_info.data_entry_coarse(data2)
                    case 0x26:
                        channel_info.data_entry_fine(data2)
                    case 0x07:
                        channel_info.set_volume_coarse(data2)
                    case 0x27:
                        channel_info.set_volume_fine(data2)
                    case 0x0A:
                        channel_info.set_pan_coarse(data2)
                    case 0x2A:
                        channel_info.set_pan_fine(data2)
                    case 0x0B:
                        channel_info.set_expression_coarse(data2)
                    case 0x2B:
                        channel_info.set_expression_fine(data2)
                    case 0x40:
                        channel_info.set_hold_pedal(data2)
                    case 0x5B:
                        channel_info.set_reverb_send(data2)
                    case 0x5D:
                        channel_info.set_chorus_send(data2)
                    case 0x65:
                        channel_info.set_rpn_coarse(data2)
                    case 0x64:
                        channel_info.set_rpn_fine(data2)
                    case 0x78:
                        self.note_off_all_channel(channel, True)
                    case 0x79:
                        self.reset_all_controllers_channel(channel)
                    case 0x7B:
                        self.note_off_all_channel(channel, False)
                    case _:
                        pass
            case 0xC0:
                channel_info.set_patch(data1)
            case 0xE0:
                channel_info.set_pitch_bend(data1, data2)
            case _:
                pass

    def note_off(self, channel: int, key: int) -> None:
        if not (0 <= channel and channel < len(self._channels)):
            return
        for i in range(self._voices.active_voice_count):
            voice = self._voices._voices[i]
            if voice.channel == channel and voice.key == key:
                voice.end()

    def note_on(self, channel: int, key: int, velocity: int) -> None:
        if velocity == 0:
            self.note_off(channel, key)
            return
        if not (0 <= channel and channel < len(self._channels)):
            return
        channel_info = self._channels[channel]
        preset_id = (channel_info.bank_number << 16) | channel_info.patch_number
        preset = self._sound_font.presets[0]
        if preset_id in self._preset_lookup:
            preset = self._preset_lookup[preset_id]
        else:
            gm_preset_id = (
                channel_info.patch_number
                if channel_info.bank_number < 128
                else (128 << 16)
            )
            if gm_preset_id in self._preset_lookup:
                preset = self._preset_lookup[gm_preset_id]
            else:
                preset = self._default_preset

        for preset_region in preset.regions:
            if preset_region.contains(key, velocity):
                for instrument_region in preset_region.instrument.regions:
                    if instrument_region.contains(key, velocity):
                        region_pair = RegionPair(preset_region, instrument_region)
                        voice = self._voices.request_new(
                            instrument_region, channel
                        )
                        voice.start(region_pair, channel, key, velocity)

    def note_off_all(self, immediate: bool) -> None:
        if immediate:
            self._voices.clear()
        else:
            for i in range(self._voices.active_voice_count):
                self._voices._voices[i].end()

    def note_off_all_channel(self, channel: int, immediate: bool) -> None:
        if immediate:
            for i in range(self._voices.active_voice_count):
                if self._voices._voices[i].channel == channel:
                    self._voices._voices[i].kill()
        else:
            for i in range(self._voices.active_voice_count):
                if self._voices._voices[i].channel == channel:
                    self._voices._voices[i].end()

    def reset_all_controllers(self) -> None:
        for channel in self._channels:
            channel.reset_all_controllers()

    def reset_all_controllers_channel(self, channel: int) -> None:
        if not (0 <= channel and channel < len(self._channels)):
            return
        self._channels[channel].reset_all_controllers()

    def reset(self) -> None:
        self._voices.clear()
        for channel in self._channels:
            channel.reset()
        self._block_read = self._block_size

    def render(
        self,
        left: MutableSequence[float],
        right: MutableSequence[float],
        offset: Optional[int] = None,
        count: Optional[int] = None,
    ) -> None:
        """Render audio into left/right buffers. Raises MeltysynthError if buffer lengths differ; ValueError if offset is set without count."""
        if len(left) != len(right):
            raise MeltysynthError(
                "The output buffers for the left and right must be the same length."
            )
        if offset is None:
            offset = 0
        elif count is None:
            raise ValueError("'count' must be set if 'offset' is set.")
        if count is None:
            count = len(left)

        wrote = 0
        while wrote < count:
            if self._block_read == self._block_size:
                self._render_block()
                self._block_read = 0
            src_rem = self._block_size - self._block_read
            dst_rem = count - wrote
            rem = min(src_rem, dst_rem)
            for t in range(rem):
                left[offset + wrote + t] = self._block_left[
                    self._block_read + t
                ]
                right[offset + wrote + t] = self._block_right[
                    self._block_read + t
                ]
            self._block_read += rem
            wrote += rem

    def _render_block(self) -> None:
        self._voices.process()
        for t in range(self._block_size):
            self._block_left[t] = 0
            self._block_right[t] = 0
        for i in range(self._voices.active_voice_count):
            voice = self._voices._voices[i]
            previous_gain_left = (
                self._master_volume * voice.previous_mix_gain_left
            )
            current_gain_left = (
                self._master_volume * voice.current_mix_gain_left
            )
            self._write_block(
                previous_gain_left,
                current_gain_left,
                voice.block,
                self._block_left,
            )
            previous_gain_right = (
                self._master_volume * voice.previous_mix_gain_right
            )
            current_gain_right = (
                self._master_volume * voice.current_mix_gain_right
            )
            self._write_block(
                previous_gain_right,
                current_gain_right,
                voice.block,
                self._block_right,
            )

    def _write_block(
        self,
        previous_gain: float,
        current_gain: float,
        source: Sequence[float],
        destination: MutableSequence[float],
    ) -> None:
        if max(previous_gain, current_gain) < SoundFontMath.non_audible():
            return
        if abs(current_gain - previous_gain) < 1.0e-3:
            ArrayMath.multiply_add(current_gain, source, destination)
        else:
            step = self._inverse_block_size * (current_gain - previous_gain)
            ArrayMath.multiply_add_slope(
                previous_gain, step, source, destination
            )

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def maximum_polyphony(self) -> int:
        return self._maximum_polyphony

    @property
    def channel_count(self) -> int:
        return Synthesizer._CHANNEL_COUNT

    @property
    def percussion_channel(self) -> int:
        return Synthesizer._PERCUSSION_CHANNEL

    @property
    def sound_font(self) -> SoundFont:
        return self._sound_font

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def active_voice_count(self) -> int:
        return self._voices.active_voice_count

    @property
    def master_volume(self) -> float:
        return self._master_volume

    @master_volume.setter
    def master_volume(self, value: float) -> None:
        self._master_volume = value
