from io import BufferedIOBase
from typing import Sequence

from pygmu2.meltysynth.io.binary_reader import BinaryReaderEx
from pygmu2.meltysynth.model.types import SampleType


class SampleHeader:
    def __init__(self, reader: BufferedIOBase) -> None:
        self._name = BinaryReaderEx.read_fixed_length_string(reader, 20)
        self._start = BinaryReaderEx.read_int32(reader)
        self._end = BinaryReaderEx.read_int32(reader)
        self._start_loop = BinaryReaderEx.read_int32(reader)
        self._end_loop = BinaryReaderEx.read_int32(reader)
        self._sample_rate = BinaryReaderEx.read_int32(reader)
        self._original_pitch = BinaryReaderEx.read_uint8(reader)
        self._pitch_correction = BinaryReaderEx.read_int8(reader)
        self._link = BinaryReaderEx.read_uint16(reader)
        self._sample_type = SampleType(BinaryReaderEx.read_uint16(reader))

    @staticmethod
    def _read_from_chunk(
        reader: BufferedIOBase, size: int
    ) -> Sequence["SampleHeader"]:
        if int(size % 46) != 0:
            raise Exception("The sample header list is invalid.")

        count = int(size / 46) - 1
        headers: list[SampleHeader] = []

        for _ in range(count):
            headers.append(SampleHeader(reader))

        SampleHeader(reader)
        return headers

    @property
    def name(self) -> str:
        return self._name

    @property
    def start(self) -> int:
        return self._start

    @property
    def end(self) -> int:
        return self._end

    @property
    def start_loop(self) -> int:
        return self._start_loop

    @property
    def end_loop(self) -> int:
        return self._end_loop

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def original_pitch(self) -> int:
        return self._original_pitch

    @property
    def pitch_correction(self) -> int:
        return self._pitch_correction
