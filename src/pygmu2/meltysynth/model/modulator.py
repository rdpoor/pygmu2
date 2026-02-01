import io
from io import BufferedIOBase

from pygmu2.meltysynth.io.binary_reader import BinaryReaderEx


class Modulator:
    @staticmethod
    def discard_data(reader: BufferedIOBase, size: int) -> None:
        if int(size % 10) != 0:
            raise Exception("The modulator list is invalid.")
        reader.seek(size, io.SEEK_CUR)
