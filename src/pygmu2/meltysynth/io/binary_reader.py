from io import BufferedIOBase
from typing import Sequence


class BinaryReaderEx:
    @staticmethod
    def read_int32(reader: BufferedIOBase) -> int:
        return int.from_bytes(reader.read(4), byteorder="little", signed=True)

    @staticmethod
    def read_uint32(reader: BufferedIOBase) -> int:
        return int.from_bytes(reader.read(4), byteorder="little", signed=False)

    @staticmethod
    def read_int16(reader: BufferedIOBase) -> int:
        return int.from_bytes(reader.read(2), byteorder="little", signed=True)

    @staticmethod
    def read_uint16(reader: BufferedIOBase) -> int:
        return int.from_bytes(reader.read(2), byteorder="little", signed=False)

    @staticmethod
    def read_int8(reader: BufferedIOBase) -> int:
        return int.from_bytes(reader.read(1), byteorder="little", signed=True)

    @staticmethod
    def read_uint8(reader: BufferedIOBase) -> int:
        return int.from_bytes(reader.read(1), byteorder="little", signed=False)

    @staticmethod
    def read_int32_big_endian(reader: BufferedIOBase) -> int:
        return int.from_bytes(reader.read(4), byteorder="big", signed=True)

    @staticmethod
    def read_int16_big_endian(reader: BufferedIOBase) -> int:
        return int.from_bytes(reader.read(2), byteorder="big", signed=True)

    @staticmethod
    def read_int_variable_length(reader: BufferedIOBase) -> int:
        acc = 0
        count = 0
        while True:
            value = BinaryReaderEx.read_uint8(reader)
            acc = (acc << 7) | (value & 127)
            if (value & 128) == 0:
                break
            count += 1
            if count == 4:
                raise Exception(
                    "The length of the value must be equal to or less than 4."
                )
        return acc

    @staticmethod
    def read_four_cc(reader: BufferedIOBase) -> str:
        data = bytearray(reader.read(4))
        for i, value in enumerate(data):
            if not (32 <= value and value <= 126):
                data[i] = 63  # '?'
        return data.decode("ascii")

    @staticmethod
    def read_fixed_length_string(reader: BufferedIOBase, length: int) -> str:
        data = reader.read(length)
        actualLength = 0
        for value in data:
            if value == 0:
                break
            actualLength += 1
        return data[0:actualLength].decode("ascii")

    @staticmethod
    def read_int16_array_as_float_array(
        reader: BufferedIOBase, size: int
    ) -> Sequence[float]:
        from array import array

        count = int(size / 2)
        data = array("h")
        data.fromfile(reader, count)
        return array("f", map(lambda x: x / 32768.0, data))
