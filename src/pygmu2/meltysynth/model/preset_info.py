from io import BufferedIOBase
from typing import Sequence

from pygmu2.meltysynth.io.binary_reader import BinaryReaderEx


class PresetInfo:
    def __init__(self, reader: BufferedIOBase) -> None:
        self._name = BinaryReaderEx.read_fixed_length_string(reader, 20)
        self._patch_number = BinaryReaderEx.read_uint16(reader)
        self._bank_number = BinaryReaderEx.read_uint16(reader)
        self._zone_start_index = BinaryReaderEx.read_uint16(reader)
        self._library = BinaryReaderEx.read_int32(reader)
        self._genre = BinaryReaderEx.read_int32(reader)
        self._morphology = BinaryReaderEx.read_int32(reader)

    @staticmethod
    def read_from_chunk(
        reader: BufferedIOBase, size: int
    ) -> Sequence["PresetInfo"]:
        if int(size % 38) != 0:
            raise MeltysynthError("The preset list is invalid.")

        count = int(size / 38)
        presets: list[PresetInfo] = []

        for i in range(count):
            presets.append(PresetInfo(reader))

        for i in range(count - 1):
            presets[i]._zone_end_index = (
                presets[i + 1].zone_start_index - 1
            )

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
    def zone_start_index(self) -> int:
        return self._zone_start_index

    @property
    def zone_end_index(self) -> int:
        return self._zone_end_index

    @property
    def library(self) -> int:
        return self._library

    @property
    def genre(self) -> int:
        return self._genre

    @property
    def morphology(self) -> int:
        return self._morphology
