from io import BufferedIOBase
from typing import Sequence

from pygmu2.meltysynth.io.binary_reader import BinaryReaderEx
from pygmu2.meltysynth.model.types import GeneratorType


class Generator:
    def __init__(self, reader: BufferedIOBase) -> None:
        self._generator_type = GeneratorType(BinaryReaderEx.read_uint16(reader))
        self._value = BinaryReaderEx.read_int16(reader)

    @staticmethod
    def read_from_chunk(
        reader: BufferedIOBase, size: int
    ) -> Sequence["Generator"]:
        if int(size % 4) != 0:
            raise Exception("The generator list is invalid.")

        count = int(size / 4) - 1
        generators: list[Generator] = []

        for _ in range(count):
            generators.append(Generator(reader))

        Generator(reader)
        return generators

    @property
    def generator_type(self) -> GeneratorType:
        return self._generator_type

    @property
    def value(self) -> int:
        return self._value
