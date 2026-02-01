from io import BufferedIOBase
from typing import Sequence

from pygmu2.meltysynth.exceptions import MeltysynthError
from pygmu2.meltysynth.io.binary_reader import BinaryReaderEx


class ZoneInfo:
    def __init__(self, reader: BufferedIOBase) -> None:
        self._generator_index = BinaryReaderEx.read_uint16(reader)
        self._modulator_index = BinaryReaderEx.read_uint16(reader)

    @staticmethod
    def read_from_chunk(
        reader: BufferedIOBase, size: int
    ) -> Sequence["ZoneInfo"]:
        if int(size % 4) != 0:
            raise MeltysynthError("The zone list is invalid.")

        count = int(size / 4)
        zones: list[ZoneInfo] = []

        for i in range(count):
            zones.append(ZoneInfo(reader))

        for i in range(count - 1):
            zones[i]._generator_count = (
                zones[i + 1]._generator_index - zones[i]._generator_index
            )
            zones[i]._modulator_count = (
                zones[i + 1]._modulator_index - zones[i]._modulator_index
            )

        return zones

    @property
    def generator_index(self) -> int:
        return self._generator_index

    @property
    def modulator_index(self) -> int:
        return self._modulator_index

    @property
    def generator_count(self) -> int:
        return self._generator_count

    @property
    def modulator_count(self) -> int:
        return self._modulator_count
