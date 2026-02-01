from enum import IntEnum


class MidiMessageType(IntEnum):
    NORMAL = 0
    TEMPO_CHANGE = 252
    END_OF_TRACK = 255


class MidiMessage:
    def __init__(
        self, channel: int, command: int, data1: int, data2: int
    ) -> None:
        self._data = bytearray()
        self._data.append(channel & 0xFF)
        self._data.append(command & 0xFF)
        self._data.append(data1 & 0xFF)
        self._data.append(data2 & 0xFF)

    @staticmethod
    def common1(status: int, data1: int) -> "MidiMessage":
        channel = status & 0x0F
        command = status & 0xF0
        data2 = 0
        return MidiMessage(channel, command, data1, data2)

    @staticmethod
    def common2(
        status: int, data1: int, data2: int
    ) -> "MidiMessage":
        channel = status & 0x0F
        command = status & 0xF0
        return MidiMessage(channel, command, data1, data2)

    @staticmethod
    def tempo_change(tempo: int) -> "MidiMessage":
        command = tempo >> 16
        data1 = tempo >> 8
        data2 = tempo
        return MidiMessage(MidiMessageType.TEMPO_CHANGE, command, data1, data2)

    @staticmethod
    def end_of_track() -> "MidiMessage":
        return MidiMessage(MidiMessageType.END_OF_TRACK, 0, 0, 0)

    @property
    def type(self) -> MidiMessageType:
        match self.channel:
            case int(MidiMessageType.TEMPO_CHANGE):
                return MidiMessageType.TEMPO_CHANGE
            case int(MidiMessageType.END_OF_TRACK):
                return MidiMessageType.END_OF_TRACK
            case _:
                return MidiMessageType.NORMAL

    @property
    def channel(self) -> int:
        return self._data[0]

    @property
    def command(self) -> int:
        return self._data[1]

    @property
    def data1(self) -> int:
        return self._data[2]

    @property
    def data2(self) -> int:
        return self._data[3]

    @property
    def tempo(self) -> float:
        return 60000000.0 / (
            (self.command << 16) | (self.data1 << 8) | self.data2
        )
