from io import BufferedIOBase
from typing import Sequence

from pygmu2.meltysynth.io.binary_reader import BinaryReaderEx


class InstrumentInfo:
    def __init__(self, reader: BufferedIOBase) -> None:
        self._name = BinaryReaderEx.read_fixed_length_string(reader, 20)
        self._zone_start_index = BinaryReaderEx.read_uint16(reader)

    @staticmethod
    def read_from_chunk(
        reader: BufferedIOBase, size: int
    ) -> Sequence["InstrumentInfo"]:
        if int(size % 22) != 0:
            raise Exception("The instrument list is invalid.")

        count = int(size / 22)
        instruments: list[InstrumentInfo] = []

        for i in range(count):
            instruments.append(InstrumentInfo(reader))

        for i in range(count - 1):
            instruments[i]._zone_end_index = (
                instruments[i + 1]._zone_start_index - 1
            )

        return instruments

    @property
    def name(self) -> str:
        return self._name

    @property
    def zone_start_index(self) -> int:
        return self._zone_start_index

    @property
    def zone_end_index(self) -> int:
        return self._zone_end_index
