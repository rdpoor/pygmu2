from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pygmu2.meltysynth.synth.synthesizer import Synthesizer


class Channel:
    def __init__(
        self, synthesizer: "Synthesizer", is_percussion_channel: bool
    ) -> None:
        self._synthesizer = synthesizer
        self._is_percussion_channel = is_percussion_channel
        self.reset()

    def reset(self) -> None:
        self._bank_number = 128 if self._is_percussion_channel else 0
        self._patch_number = 0
        self._modulation = 0
        self._volume = 100 << 7
        self._pan = 64 << 7
        self._expression = 127 << 7
        self._hold_pedal = False
        self._reverb_send = 40
        self._chorus_send = 0
        self._rpn = -1
        self._pitch_bend_range = 2 << 7
        self._coarse_tune = 0
        self._fine_tune = 8192
        self._pitch_bend = 0

    def reset_all_controllers(self) -> None:
        self._modulation = 0
        self._expression = 127 << 7
        self._hold_pedal = False
        self._rpn = -1
        self._pitch_bend = 0

    def set_bank(self, value: int) -> None:
        self._bank_number = value
        if self._is_percussion_channel:
            self._bank_number += 128

    def set_patch(self, value: int) -> None:
        self._patch_number = value

    def set_modulation_coarse(self, value: int) -> None:
        self._modulation = (self._modulation & 0x7F) | (value << 7)

    def set_modulation_fine(self, value: int) -> None:
        self._modulation = (self._modulation & 0xFF80) | value

    def set_volume_coarse(self, value: int) -> None:
        self._volume = (self._volume & 0x7F) | (value << 7)

    def set_volume_fine(self, value: int) -> None:
        self._volume = (self._volume & 0xFF80) | value

    def set_pan_coarse(self, value: int) -> None:
        self._pan = (self._pan & 0x7F) | (value << 7)

    def set_pan_fine(self, value: int) -> None:
        self._pan = (self._pan & 0xFF80) | value

    def set_expression_coarse(self, value: int) -> None:
        self._expression = (self._expression & 0x7F) | (value << 7)

    def set_expression_fine(self, value: int) -> None:
        self._expression = (self._expression & 0xFF80) | value

    def set_hold_pedal(self, value: int) -> None:
        self._hold_pedal = value >= 64

    def set_reverb_send(self, value: int) -> None:
        self._reverb_send = value

    def set_chorus_send(self, value: int) -> None:
        self._chorus_send = value

    def set_rpn_coarse(self, value: int) -> None:
        self._rpn = (self._rpn & 0x7F) | (value << 7)

    def set_rpn_fine(self, value: int) -> None:
        self._rpn = (self._rpn & 0xFF80) | value

    def data_entry_coarse(self, value: int) -> None:
        match self._rpn:
            case 0:
                self._pitch_bend_range = (self._pitch_bend_range & 0x7F) | (
                    value << 7
                )
            case 1:
                self._fine_tune = (self._fine_tune & 0x7F) | (value << 7)
            case 2:
                self._coarse_tune = value - 64
            case _:
                pass

    def data_entry_fine(self, value: int) -> None:
        match self._rpn:
            case 0:
                self._pitch_bend_range = (self._pitch_bend_range & 0xFF80) | value
            case 1:
                self._fine_tune = (self._fine_tune & 0xFF80) | value
            case _:
                pass

    def set_pitch_bend(self, value1: int, value2: int) -> None:
        self._pitch_bend = (1.0 / 8192.0) * ((value1 | (value2 << 7)) - 8192)

    @property
    def is_percussion_channel(self) -> bool:
        return self._is_percussion_channel

    @property
    def bank_number(self) -> int:
        return self._bank_number

    @property
    def patch_number(self) -> int:
        return self._patch_number

    @property
    def modulation(self) -> float:
        return (50.0 / 16383.0) * self._modulation

    @property
    def volume(self) -> float:
        return (1.0 / 16383.0) * self._volume

    @property
    def pan(self) -> float:
        return (100.0 / 16383.0) * self._pan - 50.0

    @property
    def expression(self) -> float:
        return (1.0 / 16383.0) * self._expression

    @property
    def hold_pedal(self) -> float:
        return self._hold_pedal

    @property
    def reverb_send(self) -> float:
        return (1.0 / 127.0) * self._reverb_send

    @property
    def chorus_send(self) -> float:
        return (1.0 / 127.0) * self._chorus_send

    @property
    def pitch_bend_range(self) -> float:
        return (self._pitch_bend_range >> 7) + 0.01 * (
            self._pitch_bend_range & 0x7F
        )

    @property
    def tune(self) -> float:
        return self._coarse_tune + (1.0 / 8192.0) * (self._fine_tune - 8192)

    @property
    def pitch_bend(self) -> float:
        return self.pitch_bend_range * self._pitch_bend
