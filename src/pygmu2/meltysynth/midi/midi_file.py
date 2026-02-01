import io
import itertools
from io import BufferedIOBase
from typing import Sequence

from pygmu2.meltysynth.exceptions import MeltysynthError
from pygmu2.meltysynth.io.binary_reader import BinaryReaderEx
from pygmu2.meltysynth.midi.message import MidiMessage, MidiMessageType

_INITIAL_MIN_TICK = 10**10


class MidiFile:
    def __init__(self, reader: BufferedIOBase) -> None:
        chunk_type = BinaryReaderEx.read_four_cc(reader)
        if chunk_type != "MThd":
            raise Exception(
                "The chunk type must be 'MThd', but was '"
                + chunk_type
                + "'."
            )

        size = BinaryReaderEx.read_int32_big_endian(reader)
        if size != 6:
            raise Exception("The MThd chunk has invalid data.")

        format = BinaryReaderEx.read_int16_big_endian(reader)
        if not (format == 0 or format == 1):
            raise Exception("The format " + str(format) + " is not supported.")

        self._track_count = BinaryReaderEx.read_int16_big_endian(reader)
        self._resolution = BinaryReaderEx.read_int16_big_endian(reader)

        message_lists: list[list[MidiMessage]] = []
        tick_lists: list[list[int]] = []

        for _ in range(self._track_count):
            message_list, tick_list = MidiFile._read_track(reader)
            message_lists.append(message_list)
            tick_lists.append(tick_list)

        messages, times = MidiFile._merge_tracks(
            message_lists, tick_lists, self._resolution
        )
        self._messages = messages
        self._times = times

    @classmethod
    def from_file(cls, file_path: str) -> "MidiFile":
        with open(file_path, "rb") as f:
            return cls(f)

    @staticmethod
    def _read_track(
        reader: BufferedIOBase,
    ) -> tuple[list[MidiMessage], list[int]]:
        chunk_type = BinaryReaderEx.read_four_cc(reader)
        if chunk_type != "MTrk":
            raise MeltysynthError(
                f"The chunk type must be 'MTrk', but was '{chunk_type}'."
            )

        end = BinaryReaderEx.read_int32_big_endian(reader)
        end += reader.tell()

        messages: list[MidiMessage] = []
        ticks: list[int] = []

        tick = 0
        last_status = 0

        while True:
            delta = BinaryReaderEx.read_int_variable_length(reader)
            first = BinaryReaderEx.read_uint8(reader)

            tick += delta

            if (first & 128) == 0:
                command = last_status & 0xF0
                if command == 0xC0 or command == 0xD0:
                    messages.append(MidiMessage.common1(last_status, first))
                    ticks.append(tick)
                else:
                    data2 = BinaryReaderEx.read_uint8(reader)
                    messages.append(
                        MidiMessage.common2(last_status, first, data2)
                    )
                    ticks.append(tick)
                continue

            match first:
                case 0xF0:
                    MidiFile.discard_data(reader)
                case 0xF7:
                    MidiFile.discard_data(reader)
                case 0xFF:
                    match BinaryReaderEx.read_uint8(reader):
                        case 0x2F:
                            BinaryReaderEx.read_uint8(reader)
                            messages.append(MidiMessage.end_of_track())
                            ticks.append(tick)
                            if reader.tell() < end:
                                reader.seek(end, io.SEEK_SET)
                            return messages, ticks
                        case 0x51:
                            messages.append(
                                MidiMessage.tempo_change(
                                    MidiFile.read_tempo(reader)
                                )
                            )
                            ticks.append(tick)
                        case _:
                            MidiFile.discard_data(reader)
                case _:
                    command = first & 0xF0
                    if command == 0xC0 or command == 0xD0:
                        data1 = BinaryReaderEx.read_uint8(reader)
                        messages.append(MidiMessage.common1(first, data1))
                        ticks.append(tick)
                    else:
                        data1 = BinaryReaderEx.read_uint8(reader)
                        data2 = BinaryReaderEx.read_uint8(reader)
                        messages.append(
                            MidiMessage.common2(first, data1, data2)
                        )
                        ticks.append(tick)
                    last_status = first

    @staticmethod
    def _merge_tracks(
        message_lists: list[list[MidiMessage]],
        tick_lists: list[list[int]],
        resolution: int,
    ) -> tuple[list[MidiMessage], list[float]]:
        merged_messages: list[MidiMessage] = []
        merged_times: list[float] = []
        indices = list(itertools.repeat(0, len(message_lists)))
        current_tick: int = 0
        current_time: float = 0.0
        tempo: float = 120.0

        while True:
            min_tick = _INITIAL_MIN_TICK
            min_index = -1

            for ch in range(len(tick_lists)):
                if indices[ch] < len(tick_lists[ch]):
                    tick = tick_lists[ch][indices[ch]]
                    if tick < min_tick:
                        min_tick = tick
                        min_index = ch

            if min_index == -1:
                break

            next_tick = tick_lists[min_index][indices[min_index]]
            delta_tick = next_tick - current_tick
            delta_time = 60.0 / (resolution * tempo) * delta_tick
            current_tick += delta_tick
            current_time += delta_time

            message = message_lists[min_index][indices[min_index]]
            if message.type == MidiMessageType.TEMPO_CHANGE:
                tempo = message.tempo
            else:
                merged_messages.append(message)
                merged_times.append(current_time)

            indices[min_index] += 1

        return merged_messages, merged_times

    @staticmethod
    def read_tempo(reader: BufferedIOBase) -> int:
        size = BinaryReaderEx.read_int_variable_length(reader)
        if size != 3:
            raise MeltysynthError("Failed to read the tempo value.")
        b1 = BinaryReaderEx.read_uint8(reader)
        b2 = BinaryReaderEx.read_uint8(reader)
        b3 = BinaryReaderEx.read_uint8(reader)
        return (b1 << 16) | (b2 << 8) | b3

    @staticmethod
    def discard_data(reader: BufferedIOBase) -> None:
        size = BinaryReaderEx.read_int_variable_length(reader)
        reader.seek(size, io.SEEK_CUR)

    @property
    def length(self) -> float:
        return self._times[-1]
